'''Example of VAE on MNIST dataset using CNN
The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
'''

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda, BatchNormalization, MaxPooling2D, AveragePooling2D, UpSampling2D, AveragePooling2D
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from AdamW import AdamW
from utils import psnr, load_parameters

from test_loss import median_mse_wrapper, masked_mse_wrapper, masked_binary_crossentropy

# define callback to change the value of beta at each epoch
beta = K.variable(value=0.0)
def warmup(epoch):
    value = (epoch/10.0) * (epoch <= 10.0) + 1.0 * (epoch > 10.0)
    print("\nbeta:", value)
    K.set_value(beta, value)

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def build_vae(img_shape=(28, 28, 1), latent_size=2, 
        opt='rmsprop', loss='binary_crossentropy', 
        batch_size=128, conv_layers=3, initial_filters=4):
    
    # network parameters
    h, w, ch = img_shape
    batch_size = 128
    kernel_size = 4
    filters = initial_filters
    latent_dim = latent_size
    

    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=img_shape, name='encoder_input')
    mask_inputs = Input(shape=img_shape, name='median_input')
    x = inputs
    for i in range(conv_layers):
        #filters = initial_filters if i < conv_layers-1 else 1
        conv_lyr = Conv2D(filters=initial_filters,
                kernel_size=kernel_size,
                activation='relu',
                strides=1,
                padding='same', kernel_initializer='glorot_normal', bias_initializer='zeros')
        x = conv_lyr(x)
        conv_lyr = Conv2D(filters=initial_filters,
                kernel_size=kernel_size,
                activation='relu',
                strides=2,
                padding='same', kernel_initializer='glorot_normal', bias_initializer='zeros')
        x = conv_lyr(x)
        mp = conv_lyr
        #x = BatchNormalization()(x)
        #mp = AveragePooling2D((2,2), padding='same')
        #x = MaxPooling2D((2,2), padding='same')(x)
        #x = mp(x)

    # shape info needed to build decoder model
    shape = K.int_shape(x)
    print(shape)
    #input()
    encoded_shape = x.get_shape().as_list()
    intermediate_size = encoded_shape[1]*encoded_shape[2]*encoded_shape[3]
    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(intermediate_size, activation='relu')(x)
    z_mean = Dense(latent_size, name='z_mean')(x)
    z_log_var = Dense(latent_size, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_size,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder') # -- if three outputs are needed
    encoder.summary()
    plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_size,), name='z_input')
    x = Dense(intermediate_size, activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    for i in range(conv_layers):
        filters = initial_filters #if i < conv_layers-1 else 3
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=1,
                            padding='same', kernel_initializer='glorot_normal', bias_initializer='zeros')(x)
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=2,
                            padding='same', kernel_initializer='glorot_normal', bias_initializer='zeros')(x)
        #x = BatchNormalization()(x)
        #x = UpSampling2D((2,2))(x)
        #filters = initial_filters if i < conv_layers-1 else 3

    outputs = Conv2DTranspose(filters=ch,
                            kernel_size=kernel_size,
                            activation='linear' if not 'bin-xent' in loss else 'sigmoid', 
                            padding='same',
                            name='decoder_output', kernel_initializer='glorot_normal', bias_initializer='zeros')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    plot_model(decoder, to_file='tmp/vae_cnn_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    if loss == 'wmse':
        vae = Model([inputs, mask_inputs], outputs, name='vae')
    elif loss == 'mse':
        vae = Model(inputs, outputs, name='vae')
    '''
    def vae_loss(x, x_decoded_mean):
        xent_loss = image_size * image_size * binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss
    '''
    

    def vae_r_loss(y_true, y_pred):
        ######## y_true.shape = (batch size, 3, 64, 64)

        # y_true_flat = K.flatten(y_true)
        # y_pred_flat = K.flatten(y_pred)

        # MSE
        r_loss = K.mean(K.square(y_pred - y_true), axis=[1,2,3])
        #r_loss = K.sum(K.square(y_true - y_pred), axis = [1,2,3])
        return r_loss

    LEARNING_RATE = 0.001
    KL_TOLERANCE = 0.5
    def vae_kl_loss(y_true, y_pred):

        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = 1)
        kl_loss = K.maximum(kl_loss, KL_TOLERANCE * latent_size) # 
        return kl_loss

    def vae_loss(y_true, y_pred):
        return vae_r_loss(y_true, y_pred) + beta*vae_kl_loss(y_true, y_pred)
    
    def vae_masked_entropy_loss(y_mask):
        def masked_vae(y_true, y_pred):
            y_true *= y_mask
            y_pred *= y_mask
            return binary_crossentropy(K.flatten(y_mask), K.flatten(y_pred)) + beta*vae_kl_loss(y_mask, y_pred)
        return masked_vae

    def vae_entropy_loss(y_true, y_pred):
        return binary_crossentropy(y_true, y_pred) + beta*vae_kl_loss(y_true, y_pred)

    def vae_mse_loss(y_true, y_pred):
        return vae_r_loss(y_true, y_pred) + beta*vae_kl_loss(y_true, y_pred) # 

    def vae_masked_mse_wrapper(y_mask):
        def masked_vae(y_true, y_pred):
            #y_true *= y_mask
            #y_pred *= y_mask
            return vae_r_loss(y_true * y_mask, y_pred * y_mask) + beta*vae_kl_loss(y_true, y_pred) # 
        return masked_vae

    if opt == 'adam':
        opt = Adam(lr=LEARNING_RATE)
    if opt == 'adamw':
        parameters_filepath = "config.ini"
        parameters = load_parameters(parameters_filepath)
        num_epochs = int(parameters["hyperparam"]["num_epochs"])
        batch_size = int(parameters["hyperparam"]["batch_size"])

        opt = AdamW(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., weight_decay=0.025, batch_size=batch_size, samples_per_epoch=1000, epochs=num_epochs)

    # VAE loss = mse_loss or xent_loss + kl_loss
    if loss == 'wmse':
        vae.compile(optimizer=opt, loss = vae_masked_mse_wrapper(mask_inputs),  metrics = [vae_r_loss, vae_kl_loss, psnr])
    elif loss == 'mse':
        vae.compile(optimizer=opt, loss = vae_mse_loss,  metrics = [vae_mse_loss, vae_kl_loss, psnr])
    elif loss == 'binary_crossentropy' or loss == 'bin-xent':
        vae.compile(optimizer=opt, loss = vae_entropy_loss,  metrics = [vae_r_loss, vae_kl_loss, psnr])
    elif loss == 'weighted_binary_crossentropy' or loss == 'wbin-xent':
        vae.compile(optimizer=opt, loss = vae_masked_entropy_loss(mask_inputs),  metrics = [vae_r_loss, vae_kl_loss, psnr])
    else:
        raise NotImplementedError

    #vae_full.compile(optimizer=opti, loss = vae_loss,  metrics = [vae_r_loss, vae_kl_loss])

    '''
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss) 
    vae.metrics = [psnr]
    vae.add_loss(vae_loss)
    vae.compile(optimizer=opt, loss='', metrics=[psnr])
    '''
    #vae.compile(optimizer=opt, loss = vae_masked_mse_wrapper(mask_inputs),  metrics = [vae_r_loss, vae_kl_loss, psnr])
    vae.summary()
    #plot_model(vae, to_file='tmp/vae_cnn.png', show_shapes=True)

    return  vae, encoder, decoder, encoded_shape


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()

if __name__ == '__main__':

    # laptop & dual GPU systems
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="1" # in the external slot
    #os.environ["CUDA_VISIBLE_DEVICES"]="0" # on the mobo

    # config
    num_epochs = 30
    batch_size = 128
    conv_layers = 2 

    # MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    data = (x_test, y_test)
    
    vae, encoder, decoder, latent_shape = build_vae(batch_size=batch_size, conv_layers=conv_layers)
    _, lat_h, lat_w, lat_ch = latent_shape
    experiment_label = "vae-e{}.bs{}.lat{}x{}x{}.conv{}".format(num_epochs, batch_size, lat_h, lat_w, lat_ch, conv_layers)

    models = (encoder, decoder)
    

    vae.fit(x_train,
            epochs=num_epochs,
            batch_size=batch_size,
            validation_data=(x_test, None))
    vae.save_weights('trained_models/vae_cnn_mnist.h5')

    plot_results(models, data, batch_size=batch_size, model_name="vae_cnn")