
# includes all keras layers for building the model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Reshape, Conv2DTranspose, ZeroPadding2D
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
# import regularizer
#from keras.regularizers import l2
# instantiate regularizer
#reg = l1(0.001)

from utils import psnr
from dssim import DSSIMObjective

# example of a wrapper loss - generator must output 3 values x, [y, y_p]

def MAEpw_wrapper(y_prec):
    def MAEpw(y_true, y_pred):
        return K.mean(K.square(y_prec * (y_pred - y_true)))
    return MAEpw


def weighted_mean_squared_error(y_true, y_pred, factor=100.0):
    true_mean = K.mean(y_true, axis=0) # get batch mean (average image)
    pred_mean = K.mean(y_pred, axis=0)

    #bs = K.eval(K.int_shape(y_true)[0]) ## batch size
    #median_index = tf.constant(15)
    reordered = tf.gather(y_true, tf.nn.top_k(y_true[:], k=32).indices)
    #sorted_y_true = tf.compat.s.v1.sort(y_true, axis=0)
    #median_y_true = tf.gather(sorted_y_true, median_index, axis=0)
    tmp = reordered - y_true
    #print()

    print("Mean weight loss shape: ", K.int_shape(true_mean))

    y_true = (true_mean - y_true) * factor
    y_pred = (pred_mean - y_pred) * factor

    #y_true = K.flatten(y_true)
    #y_pred = K.flatten(y_pred)

    return K.mean(K.square(y_pred - y_true), axis=-1)

def build_multi_ae():
    pass # TODO: multi input multi output ae

def build_conv_ae(img_shape=(32, 32, 3), latent_size=16, opt='adam', loss='mse', conv_layers=4, initial_filters=4):
    
    _, _, ch = img_shape
    input_img = Input(shape=img_shape)  # adapt this if using `channels_first` image data format
    filters = initial_filters
    kernel_size = (3,3)
    s = 1 # stride parameter

    x = input_img
    for i in range(conv_layers):
        filters *= 2
        conv_lyr = Conv2D(filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                strides=s,
                padding='same')
        x = conv_lyr(x)
        #x = BatchNormalization()(x)
        mp = MaxPooling2D((2,2), padding='same')
        x = mp(x)
        
        
    '''
    x = Conv2D(32, kernel_size, activation='relu', 
                                padding='same', 
                                strides=(s,s)
                                )(input_img) # 
    x = Conv2D(64, kernel_size, activation='relu', 
                                padding='same', 
                                strides=(s,s)
                                )(x) # 

    conv_lyr = Conv2D(128, kernel_size, activation='relu', padding='same', strides=(s,s)) 
    x = conv_lyr(x)
    '''
    conv_shape = mp.output_shape[1:] 
    #conv_shape = conv_lyr.output_shape[1:] #
    print(conv_shape)
    #conv_shape = mp.output_shape[1:] # without the batch_size

    #flat_lyr = GlobalAveragePooling2D() # GAP layer
    #x = flat_lyr(x)
    #flatten_dim = conv_shape[0]*conv_shape[1]*conv_shape[2] #flat_lyr.output_shape[-1]
    #gap_dim = flat_lyr.output_shape[-1]
    print("conv_shape:", conv_shape)
    flat_lyr = Flatten(name='flat_in')
    x = flat_lyr(x) # first call the layer, then it will have its shape
    flatten_dim = flat_lyr.output_shape[-1] # last entry in the tuple of f is the flattened dimension
    print(type(x), type(flat_lyr))
    print("flatten_dim: ", flatten_dim)
    encoded_layer = Dense(latent_size, activation='relu', name='latent')
    encoded = encoded_layer(x)
    end_flat_layer = Dense(flatten_dim, activation='relu', name='flat_out')
    x = end_flat_layer(encoded) # increase the dimension of data to the point before latent size (to match first Flatten layer shape)
    #x = Dense(flatten_dim)(x)

    x = Reshape(target_shape=conv_shape)(x)

    '''
    x = Conv2DTranspose(128, kernel_size, strides=(s,s), activation='relu', padding='same')(x)
    #x = BatchNormalization()(x)
    #x = UpSampling2D((2,2))(x)
    x = Conv2DTranspose(64, kernel_size, strides=(s,s), activation='relu', padding='same')(x)
    #x = BatchNormalization()(x)
    #x = UpSampling2D((2,2))(x)
    x = Conv2DTranspose(32, kernel_size, strides=(s,s), activation='relu', padding='same')(x)
    #x = BatchNormalization()(x)
    #x = UpSampling2D((2,2))(x)
    '''
    for i in range(conv_layers):
        x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=s,
                        padding='same')(x)
        #x = BatchNormalization()(x)
        x = UpSampling2D((2,2))(x)
        filters //= 2

    decoded_layer = Conv2D(ch, kernel_size, activation='linear' if loss != 'bin-xent' else 'sigmoid', padding='same')
    decoded = decoded_layer(x)

    autoencoder = Model(input_img, decoded)
    print(autoencoder.summary())
    # TODO: specify learning rate?
    #if opt == 'adam':

    if opt == 'adam':
        opt = Adam(lr=0.001) # try bigger learning rate

    if loss == 'bin-xent':
        loss = 'binary_crossentropy'
    if loss == 'dssim':
        loss = DSSIMObjective()
    if loss == 'wmse':
        loss = weighted_mean_squared_error
    autoencoder.compile(optimizer=opt, loss=loss, metrics=[psnr])

    #print(autoencoder.summary())
    #input("Press any key...")
    #print("# AE layers: ", len(autoencoder.layers))

    # create encoder model, which will be able to encode the image into latent representation
    encoder = Model(input_img, encoded)
    
    #encoded_shape = encoded.get_shape().as_list()
    #_, enc_h, enc_w, enc_ch = encoded_shape
    #enc_h, enc_w, enc_ch = 4, 4, 8
    #print("latent shape: ", latent_size)
    #print("decoded shape: ", autoencoder.layers[-8].input.shape)
    
    # re-create decoder model, which will be able to decode encoded input
    encoded_input = Input(shape=(latent_size,)) # skip batch size which is None
    #print(autoencoder.layers[-6](encoded_input))
    #deco = autoencoder.layers[-8](encoded_input)
    #deco = autoencoder.layers[-7](encoded_input)

    deco = encoded_input
    assemble = False
    for layer in autoencoder.layers:
        if layer == end_flat_layer:
            assemble = True
        if assemble:
            deco = layer(deco)

    decoded_output = deco
    '''
    deco = autoencoder.layers[-11](encoded_input)
    for i in range(10, 1):
        deco = autoencoder.layers[-i](deco)
    decoded_output = autoencoder.layers[-1](deco)
    '''
    '''
    deco = autoencoder.layers[-6](encoded_input)
    deco = autoencoder.layers[-5](deco)
    deco = autoencoder.layers[-4](deco)
    deco = autoencoder.layers[-3](deco)
    deco = autoencoder.layers[-2](deco)
    decoded_output = autoencoder.layers[-1](deco)
    '''
    decoder = Model(encoded_input, decoded_output)
    
    return autoencoder, encoder, decoder, latent_size


def build_mnist_ae(img_shape=(28, 28, 1), opt='adadelta', loss='binary_crossentropy'):
    input_img = Input(shape=img_shape)  # adapt this if using `channels_first` image data format
    h, w, ch = img_shape

    #x = Dropout(rate=0.3)(input_img)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img) # 
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x) #
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

    encoded = MaxPooling2D((2, 2), padding='same')(x)
   
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x) # 
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(ch, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    if loss == 'binxent':
        loss = 'binary_crossentropy'
    if loss == 'dssim':
        loss = DSSIMObjective()
    if loss == 'wmse':
        loss = weighted_mean_squared_error
    autoencoder.compile(optimizer=opt, loss=loss, metrics=[psnr])

    print(autoencoder.summary())
    input("Press any key...")
    print("# AE layers: ", len(autoencoder.layers))

    # create encoder model, which will be able to encode the image into latent representation
    encoder = Model(input_img, encoded)
    
    encoded_shape = encoded.get_shape().as_list()
    _, enc_h, enc_w, enc_ch = encoded_shape
    #enc_h, enc_w, enc_ch = 4, 4, 8
    print("latent shape: ",  enc_h, enc_w, enc_ch)
    print("decoded shape: ",  autoencoder.layers[-8].input.shape)
    
    # create decoder model, which will be able to decode encoded input
    encoded_input = Input(shape=(enc_h, enc_w, enc_ch)) # skip batch size which is None
    print(autoencoder.layers[-8](encoded_input))

    deco = autoencoder.layers[-8](encoded_input)
    deco = autoencoder.layers[-7](deco)
    deco = autoencoder.layers[-6](deco)
    deco = autoencoder.layers[-5](deco)
    deco = autoencoder.layers[-4](deco)
    deco = autoencoder.layers[-3](deco)
    deco = autoencoder.layers[-2](deco)
    decoded_output = autoencoder.layers[-1](deco)
    """
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded_input)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded_output = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    """
    decoder = Model(encoded_input, decoded_output) # (encoded_input)
    
    return autoencoder, encoder, decoder, encoded_shape

def Unet_linear(img_shape=(32, 32, 3), latent_size=16, opt='adam', loss='mse', conv_layers=4, initial_filters=4):
#(img_shape, num_class): 
    
    input_height,input_width,ch =img_shape
    num_class = ch
    inputs = Input(shape = img_shape)
    #inputs = Input((nChannels, input_height, input_width))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)

    up1 = layers.concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=3)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
    
    up2 = layers.concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=3)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv5)
    
    conv6 = Convolution2D(num_class, 1, 1, activation='relu',border_mode='same')(conv5)
    conv6 = core.Reshape((input_height*input_width, num_class))(conv6)
    #conv6 = core.Permute((2,1))(conv6)

    #conv6 = core.Activation('softmax')(conv6)
    conv7 = core.Reshape((input_height, input_width, num_class))(conv6)
    
    # TODO: test this if it can output splitted face / background tensors
    # this would work nicely when connecting other models to the outputs of segmentor
    '''
    outs = []
    for num_class, label_class in zip(range(num_class), ['face','back']):
        o = Lambda(lambda x : x[:,num_class,:,:])(x)
        outs.append(o)
    model = Model(input=inputs, output=outs)
    '''
    model = Model(input=inputs, output=conv7)

    return model

if __name__ == '__main__':

    autoencoder, encoder, decoder, latent_size = build_conv_ae()
    #autoencoder, encoder, decoder, latent_size = build_mnist_ae()
    autoencoder, encoder, decoder, latent_size = build_unet() 