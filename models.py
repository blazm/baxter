
# includes all keras layers for building the model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Reshape, Conv2DTranspose, ZeroPadding2D
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l1
from keras import backend as K
import tensorflow as tf
# import regularizer
#from keras.regularizers import l2
# instantiate regularizer
#reg = l1(0.001)

from utils import psnr, load_parameters
from dssim import DSSIMObjective

from AdamW import AdamW

from test_loss import median_mse_wrapper, masked_mse_wrapper, masked_binary_crossentropy

def build_multi_ae():
    pass # TODO: multi input multi output ae


def build_conv_only_ae(img_shape=(32, 32, 3), latent_size=16, opt='adam', loss='mse', conv_layers=4, initial_filters=4):
    
    _, _, ch = img_shape
    input_img = Input(shape=img_shape)  # adapt this if using `channels_first` image data format
    input_mask = Input(shape=img_shape)  # input layer for mask (it is only used in the calculation of the loss)
    filters = initial_filters
    kernel_size = (3,3)
    s = 1 # stride parameter

    x = input_img
    #x = Conv2D(1, (1,1), activation='relu', padding='same', kernel_initializer='glorot_uniform', bias_initializer='zeros')(x) # turn to grayscale 
    for i in range(conv_layers):
        filters = initial_filters if i < conv_layers-1 else 1 #*= 2
        #x = Dropout(rate=0.1)(x)
        conv_lyr = Conv2D(filters=initial_filters,
                kernel_size=kernel_size,
                activation='elu',
                strides=s,
                padding='same', kernel_initializer='glorot_normal', bias_initializer='zeros')
        x = conv_lyr(x)
        conv_lyr = Conv2D(filters=filters,
                kernel_size=kernel_size,
                activation='elu',
                strides=2,
                padding='same', kernel_initializer='glorot_normal', bias_initializer='zeros')
        x = conv_lyr(x)
        '''
        conv_lyr = Conv2D(filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                strides=s,
                padding='same')
        x = conv_lyr(x)
        '''
        mp = conv_lyr
        #x = BatchNormalization()(x)
        #mp = AveragePooling2D((2,2), padding='same')
        #mp = MaxPooling2D((2,2), padding='same')
        #x = mp(x)
        
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
    latent_size = conv_shape[0]*conv_shape[1]*conv_shape[2]
    #conv_shape = mp.output_shape[1:] # without the batch_size

    encoded_layer = mp # Dense(latent_size, activation='relu', name='latent', activity_regularizer=l1(10e-5))
    encoded = x # encoded_layer(x)
    

    for i in range(conv_layers):
        filters = initial_filters if i < conv_layers-1 else 3
        x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        activation='elu',
                        strides=2,
                        padding='same', kernel_initializer='glorot_normal', bias_initializer='zeros')(x)
        x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        activation='elu',
                        strides=s,
                        padding='same', kernel_initializer='glorot_normal', bias_initializer='zeros')(x)
        '''
        x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=s,
                        padding='same')(x)
        '''
        #x = BatchNormalization()(x)
        #x = UpSampling2D((2,2))(x)
        #filters //= 2
        
    decoded_layer = Conv2D(ch, kernel_size, 
            activation='linear' if not 'bin-xent' in loss else 'sigmoid', padding='same'
            , kernel_initializer='glorot_normal', bias_initializer='zeros')
    decoded = decoded_layer(x)

    if loss == 'wmse' or loss == 'wbin-xent':
        autoencoder = Model([input_img, input_mask], decoded)
    else:
        autoencoder = Model(input_img, decoded)
    print(autoencoder.summary())
    # TODO: specify learning rate?
    #if opt == 'adam':

    if opt == 'adam':
        opt = Adam(lr=0.001) # try bigger learning rate
    if opt == 'adamw':
        parameters_filepath = "config.ini"
        parameters = load_parameters(parameters_filepath)
        num_epochs = int(parameters["hyperparam"]["num_epochs"])
        batch_size = int(parameters["hyperparam"]["batch_size"])

        opt = AdamW(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., weight_decay=0.025, batch_size=batch_size, samples_per_epoch=1000, epochs=num_epochs)

    if loss == 'wbin-xent':
        loss = masked_binary_crossentropy(input_mask)
    elif loss == 'bin-xent':
        loss = 'binary_crossentropy'
    if loss == 'dssim':
        loss = DSSIMObjective()
    if loss == 'wmse':
        loss = masked_mse_wrapper(input_mask)

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
    encoded_input = Input(shape=conv_shape) # skip batch size which is None
    #print(autoencoder.layers[-6](encoded_input))
    #deco = autoencoder.layers[-8](encoded_input)
    #deco = autoencoder.layers[-7](encoded_input)

    deco = encoded_input
    assemble = False
    for layer in autoencoder.layers:
        if assemble:
            deco = layer(deco)
        if layer == encoded_layer:
            assemble = True
        

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


def build_conv_dense_ae(img_shape=(32, 32, 3), latent_size=16, opt='adam', loss='mse', conv_layers=4, initial_filters=4):
    
    _, _, ch = img_shape
    input_img = Input(shape=img_shape)  # adapt this if using `channels_first` image data format
    input_mask = Input(shape=img_shape) 
    filters = initial_filters
    kernel_size = (3,3)
    s = 1 # stride parameter

    x = input_img
    for i in range(conv_layers):
        #filters = initial_filters if i < conv_layers-1 else 4 #*= 2
        conv_lyr = Conv2D(filters=initial_filters,
                kernel_size=kernel_size,
                activation='elu',
                strides=s,
                padding='same', kernel_initializer='glorot_normal', bias_initializer='zeros')
        x = conv_lyr(x)
        conv_lyr = Conv2D(filters=initial_filters,
                kernel_size=kernel_size,
                activation='elu',
                strides=s,
                padding='same', kernel_initializer='glorot_normal', bias_initializer='zeros')
        x = conv_lyr(x)
        conv_lyr = Conv2D(filters=filters,
                kernel_size=kernel_size,
                activation='elu',
                strides=2,
                padding='same', kernel_initializer='glorot_normal', bias_initializer='zeros')
        x = conv_lyr(x)
        mp = conv_lyr
        
        
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
    encoded_layer = Dense(latent_size, activation='elu', name='latent',
                            kernel_initializer='glorot_normal', bias_initializer='zeros')
    encoded = encoded_layer(x)
    end_flat_layer = Dense(flatten_dim, activation='elu', name='flat_out',
                            kernel_initializer='glorot_normal', bias_initializer='zeros')
    x = end_flat_layer(encoded) # increase the dimension of data to the point before latent size (to match first Flatten layer shape)
    x = Reshape(target_shape=conv_shape)(x)

    for i in range(conv_layers):
        filters = initial_filters if i < conv_layers-1 else 3
        x = Conv2DTranspose(filters=initial_filters,
                        kernel_size=kernel_size,
                        activation='elu',
                        strides=s,
                        padding='same', kernel_initializer='glorot_normal', bias_initializer='zeros')(x)
        x = Conv2DTranspose(filters=initial_filters,
                        kernel_size=kernel_size,
                        activation='elu',
                        strides=s,
                        padding='same', kernel_initializer='glorot_normal', bias_initializer='zeros')(x)
        x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        activation='elu',
                        strides=2,
                        padding='same', kernel_initializer='glorot_normal', bias_initializer='zeros')(x)
        #x = BatchNormalization()(x)
        #x = UpSampling2D((2,2))(x)
        #filters //= 2
        
    decoded_layer = Conv2D(ch, kernel_size, 
            activation='linear' if not 'bin-xent' in loss else 'sigmoid', padding='same',
            kernel_initializer='glorot_normal', bias_initializer='zeros')
    decoded = decoded_layer(x)

    decoded_layer = Conv2D(ch, kernel_size, activation='linear' if not 'bin-xent' in loss else 'sigmoid', padding='same')
    decoded = decoded_layer(x)

    if loss == 'wmse':
        autoencoder = Model([input_img, input_mask], decoded)
    else:
        autoencoder = Model(input_img, decoded)
    print(autoencoder.summary())
    # TODO: specify learning rate?
    #if opt == 'adam':

    if opt == 'adam':
        opt = Adam(lr=0.001) # try bigger learning rate
    if opt == 'adamw':
        parameters_filepath = "config.ini"
        parameters = load_parameters(parameters_filepath)
        num_epochs = int(parameters["hyperparam"]["num_epochs"])
        batch_size = int(parameters["hyperparam"]["batch_size"])
        opt = AdamW(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., weight_decay=0.025, batch_size=batch_size, samples_per_epoch=1000, epochs=num_epochs)

    if loss == 'wbin-xent':
        loss = masked_binary_crossentropy(input_mask)
    if loss == 'bin-xent':
        loss = 'binary_crossentropy'
    if loss == 'dssim':
        loss = DSSIMObjective()
    if loss == 'wmse':
        loss = masked_mse_wrapper(input_mask)
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

if __name__ == '__main__':
    autoencoder, encoder, decoder, latent_size = build_conv_dense_ae()
    autoencoder, encoder, decoder, latent_size = build_conv_only_ae()
    autoencoder, encoder, decoder, latent_size = build_mnist_ae()