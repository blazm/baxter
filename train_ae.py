import os
#os.environ["TF_USE_CUDNN"]="0"   # set CUDNN to false, since it is non-deterministic: https://github.com/keras-team/keras/issues/2479
#os.environ['PYTHONHASHSEED'] = '0'

# ensure reproducibility
#'''
from numpy.random import seed
seed(2)
from tensorflow import set_random_seed
set_random_seed(5)
#'''

# includes all keras layers for building the model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model
from keras import backend as K

# include demo dataset mnist
from keras.datasets import mnist
import numpy as np

from scipy.misc import imresize, imsave #, imread, imsave
from scipy.ndimage import imread
from skimage.transform import resize

from utils import trim, read_files, psnr, log10, preprocess_size, loadAndResizeImages2
from models import build_conv_dense_ae, build_mnist_ae, build_conv_only_ae
from train_vae import build_vae
from world_models_vae_arch import build_vae_world_model

from data_generators import data_generator, data_generator_mnist, random_data_generator
from utils import load_parameters, list_images_recursively

def load_data(shape=(28, 28, 1)):
    h, w, ch = shape
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    #from skimage.transform import resize
    #x_train = resize(x_train, (h, w), anti_aliasing=True)
    #x_test = resize(x_test, (h, w), anti_aliasing=True)

    x_train = np.reshape(x_train, (len(x_train), h, w, ch))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), h, w, ch))  # adapt this if using `channels_first` image data format

    #x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    #x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

    return x_train, x_test

if __name__ == "__main__":

    # random generator config
    dir_with_src_images = 'data\\generated\\'
    base_image = 'median_image.png'
    object_images = ['circle.png', 'robo.png'] # circle in the first place, as robo can be drawn over it

    # default config
    img_dim = 32
    img_shape = (img_dim, img_dim, 1)
    num_epochs = 10
    batch_size = 32
    latent_size = 16 # 25, 36, 49, 64
    conv_layers = 3
    loss = 'mse' #bin-xent' #'mse' #   #
    opt = 'adadelta' #'adam' # # 
    model_label = 'ae_v2'
    do_train = False
    do_test = True
    interactive = True
    num_filters = 3
    kernel_size = 3
    kernel_mult = 1

    # TODO: load config from config.ini
    parameters_filepath = "config.ini"
    # TODO: copy parameters to separate versioned files (to archive parameters for each run)
    parameters = load_parameters(parameters_filepath)

    do_train = eval(parameters["general"]["do_train"])
    do_test = eval(parameters["general"]["do_test"])
    selected_gpu = eval(parameters["general"]["selected_gpu"])
    interactive = eval(parameters["general"]["interactive"])

    train_dir = eval(parameters["dataset"]["train_dir"])
    valid_dir = eval(parameters["dataset"]["valid_dir"])
    test_dir = eval(parameters["dataset"]["test_dir"])
    img_shape = eval(parameters["dataset"]["img_shape"])

    size_factor = float(parameters["synthetic"]["size_factor"])
    obj_attention = float(parameters["synthetic"]["obj_attention"])
    back_attention = float(parameters["synthetic"]["back_attention"])

    num_epochs = int(parameters["hyperparam"]["num_epochs"])
    batch_size = int(parameters["hyperparam"]["batch_size"])
    latent_size = int(parameters["hyperparam"]["latent_size"])
    conv_layers = int(parameters["hyperparam"]["conv_layers"])
    num_filters = int(parameters["hyperparam"]["num_filters"])
    kernel_size = int(parameters["hyperparam"]["kernel_size"])
    kernel_mult = int(parameters["hyperparam"]["kernel_mult"])
    loss = (parameters["hyperparam"]["loss"])
    opt = (parameters["hyperparam"]["opt"])
    model_label = (parameters["hyperparam"]["model_label"])

    # Blaz's laptop & other dual GPU systems
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=selected_gpu  # "1" in the external slot
    #os.environ["CUDA_VISIBLE_DEVICES"]="0" # on the mobo

    # vae from world models
    if 'vaewm' in model_label:
        autoencoder, encoder, decoder_mu_log_var, decoder, latent_shape = build_vae_world_model(
            img_shape=img_shape, latent_size=latent_size, 
            opt=opt, loss=loss, #batch_size=batch_size, 
            conv_layers=conv_layers, initial_filters=num_filters) #, kernel_size=kernel_size, kernel_mult=kernel_mult)
    # vae
    elif 'vae' in model_label:
        autoencoder, encoder, decoder, latent_shape = build_vae(
            img_shape=img_shape, latent_size=latent_size, 
            opt=opt, loss=loss, batch_size=batch_size, 
            conv_layers=conv_layers, initial_filters=num_filters) #, kernel_size=kernel_size, kernel_mult=kernel_mult)
    # ae
    elif 'ae_conv' in model_label:
        autoencoder, encoder, decoder, latent_size = build_conv_only_ae(
            img_shape=img_shape, latent_size=latent_size, 
            opt=opt, loss=loss, 
            conv_layers=conv_layers, initial_filters=num_filters) #, kernel_size=kernel_size, kernel_mult=kernel_mult)
    #autoencoder, encoder, decoder, latent_size = build_mnist_ae(img_shape=img_shape, opt=opt, loss=loss)
    elif 'ae_dense' in model_label:
        autoencoder, encoder, decoder, latent_size = build_conv_dense_ae(
            img_shape=img_shape, latent_size=latent_size, 
            opt=opt, loss=loss, 
            conv_layers=conv_layers, initial_filters=num_filters)
    
    if interactive:
        autoencoder.summary()
        input("Press any key...")
    print(latent_size)
    
    if not type(latent_size) == int:
        _, lat_h, lat_w, lat_ch = latent_size
        #lat_w *= 8
    else:
        if int(np.sqrt(latent_size))**2 == latent_size:
            lat_h, lat_w = int(np.sqrt(latent_size)), int(np.sqrt(latent_size))
            lat_ch = 1
        else:
            import math
            lat_ch = 1
            if latent_size % 3 == 0:
                lat_ch = 3

            def divisorGenerator(n):
                large_divisors = []
                for i in range(1, int(math.sqrt(n) + 1)):
                    if n % i == 0:
                        yield i
                        if i*i != n:
                            large_divisors.append(n / i)
                for divisor in reversed(large_divisors):
                    yield divisor

            tmp = list(divisorGenerator(latent_size // lat_ch))

            lat_h = int(tmp[len(tmp)//2])
            lat_w = int(latent_size // (lat_h*lat_ch))

    experiment_label = "{}.osize-{}.oatt-{}.e{}.bs{}.lat{}.c{}.opt-{}.loss-{}".format(
        model_label, size_factor, obj_attention, num_epochs, batch_size,
        "{:02d}".format(latent_size) if type(latent_size) == int else 'x'.join(map(str,latent_size[1:])), 
        conv_layers, opt, loss)

    # copy parameters into snapshots archive
    from shutil import copy2 as copyfile
    copyfile('config.ini', 'snapshots/{}.ini'.format(experiment_label))

    if train_dir is None:
        x_train, x_test = load_data()
        total_images = x_train.shape[0]
   
    if do_train:
        from keras.callbacks import TensorBoard, CSVLogger, LearningRateScheduler

        if train_dir is not None:
            train_images = list_images_recursively(train_dir)
            total_images = len(train_images)
            print("Total train images: ", total_images, train_images[0])
            #fitting_generator = data_generator(train_dir, train_images, img_shape=img_shape,  batch_size=batch_size)

            fitting_generator = random_data_generator(dir_with_src_images, base_image, object_images, img_shape=img_shape, batch_size=batch_size)
        else: # if no data, then use mnist
            # implement fit_generator (inside generator provide a mechanism to train online -- wait for new samples to arrive in the window)
            fitting_generator = data_generator_mnist(x_train, x_test, (img_dim,img_dim,1), True, batch_size)

        if valid_dir is not None:
            valid_images = list_images_recursively(valid_dir)
            print("Total valid images: ", len(valid_images), valid_images[0])
            if interactive:
                input("Press any key...")
            #valid_generator = data_generator(valid_dir, valid_images, img_shape=img_shape,  batch_size=batch_size)
            valid_generator = random_data_generator(dir_with_src_images, base_image, object_images, img_shape=img_shape, batch_size=batch_size)
        else: # if no data, then use mnist
            valid_generator = data_generator_mnist(x_train, x_test, (img_dim,img_dim,1), False, batch_size)

        tb = TensorBoard(log_dir='tf-log/{}'.format(experiment_label))
        csv = CSVLogger('snapshots/{}.csv'.format(experiment_label), append=False, separator=',')
        def exp_decay(epoch):
            initial_lrate = 0.01
            k = 0.01
            lrate = initial_lrate * np.exp(-k*epoch)
            print("Epoch: ", epoch, " LR: ", lrate)
            return lrate
        lrate = LearningRateScheduler(exp_decay)
        from train_vae import warmup
        from keras.callbacks import LambdaCallback
        wu_cb = LambdaCallback(on_epoch_end=lambda epoch, log: warmup(epoch))

        callbacks=[wu_cb, tb, csv] #  lrate, 
        steps = 1000 // batch_size #np.ceil(total_images // batch_size)
        
        history = autoencoder.fit_generator(fitting_generator, steps_per_epoch=steps, epochs=num_epochs, verbose=1, 
            callbacks=callbacks, validation_data=valid_generator, validation_steps=5)
        '''
        history = autoencoder.fit(x_train, x_train,
                        epochs=num_epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(x_test, x_test),
                        callbacks=[TensorBoard(log_dir='tf-log/{}'.format(experiment_label))])
        '''
        try:
            autoencoder.save('trained_models/{}.h5'.format(experiment_label))  
        except Exception as e:
            print("Could not save model: ", str(e))
        autoencoder.save_weights('trained_models/{}_weights.h5'.format(experiment_label))

        #plot(history)

    if do_test:

        

        autoencoder.load_weights('trained_models/{}_weights.h5'.format(experiment_label))

        def resize_dataset(dataset, img_shape=(32,32,1)):
            h, w, ch = img_shape
            data_len = dataset.shape[0]
            resized_dataset = np.zeros((data_len, h, w, ch))
            for i in range(data_len):
                resized_dataset[i] = resize(dataset[i], (h, w), anti_aliasing=True)
            return resized_dataset

        if test_dir is not None:
            test_images = list_images_recursively(valid_dir)
            print("Total valid images: ", len(test_images), test_images[0])
            if interactive:
                input("Press any key...")

            #valid_generator = data_generator(valid_dir, valid_images, img_shape=img_shape,  batch_size=batch_size, mode=generator_mode)
            valid_generator = random_data_generator(dir_with_src_images, base_image, object_images, img_shape=img_shape, batch_size=64)
            #valid_generator = data_generator(test_dir, test_images, img_shape=img_shape, batch_size=1, mode=generator_mode)
            h,w,ch = img_shape
            x_test = np.zeros((len(test_images), h, w, ch))
            
            x_mask = np.zeros((len(test_images), h, w, ch))
            i = 0
            for [img, mask], out in valid_generator:
                
                x_test[i] = img[0] #resize(img, (h, w), anti_aliasing=True)
                #print(mask[0].shape, mask[0].min(), mask[0].max())
                x_mask[i] = mask[0]
                i += 1
                if i >= len(test_images):
                    break

            # TODO: test if this returns median image n x n x 3
            back_generator = random_data_generator(dir_with_src_images, base_image, object_images, img_shape=img_shape, batch_size=64)
            [batch_inputs, batch_masks], batch_outputs = next(back_generator)
            background = np.median(batch_outputs, axis=0, keepdims=False)
            print("median image shape: ", background.shape) 
            #exit()
        else: # if no data, then use mnist
            #valid_generator = data_generator_mnist(x_train, x_test, (img_dim,img_dim,1), False, batch_size)
            x_test = resize_dataset(x_test)
            background = np.median(x_test, axis=0, keepdims=True)
            x_mask = x_test - background
            background = background[0]

        #decoded_imgs = autoencoder.predict(x_test)
        print("input images shape: ", x_test.shape)
        encoded_imgs = encoder.predict(x_test)
        if type(encoded_imgs) == list:
            encoded_imgs = encoded_imgs[-1]
        print("encoded images shape: ", encoded_imgs.shape) #, encoded_imgs[1].shape, encoded_imgs[2].shape)
        print("encoded MAX / MIN: ", encoded_imgs.max(), " / ", encoded_imgs.min())
        
        # normalize before displaying
        #encoded_imgs = (encoded_imgs - encoded_imgs.min()) / np.ptp(encoded_imgs) * 255.0
        #print(encoded_imgs)
        decoded_imgs = decoder.predict(encoded_imgs)
        print("h, w, ch: {},{},{}".format(lat_h, lat_w, lat_ch))
        print("encoded MAX / MIN: ", encoded_imgs.max(), " / ", encoded_imgs.min())
        
        print("input images shape: ", x_test.shape)
        print("decoded_imgs images shape: ", decoded_imgs.shape)
        latent_dreams = encoder.predict(decoded_imgs)
        dreams = decoder.predict(latent_dreams) # dream the images

        print("decoded images shape: ", decoded_imgs.shape)
        import matplotlib.pyplot as plt

        # TODO: evaluate the reconstructed images
        def cdist(a, b):
            return np.sqrt(np.sum(np.square(a - b), axis=2)) # axis=2 : sum by channels

        def cdist_average(avg, b):
            avg = np.tile(avg, b.shape)

        max_cdist = cdist(np.zeros((1, 1, 3)), np.ones((1, 1, 3)))
        #print("Max cdist: ", max_cdist)

        threshold = .2
        class_masks_R = np.zeros((len(test_images), h, w, ch))
        class_masks_B = np.zeros((len(test_images), h, w, ch))

        IoUs = np.zeros((len(test_images)))
        minRBs = np.zeros((len(test_images)))
        for i in range(x_test.shape[0]):
            original = x_test[i]
            mask = x_mask[i]#.astype(int)
            
            reconstructed = decoded_imgs[i]
            #print(original.max(), original.min(), reconstructed.max(), reconstructed.min())
            # pass the positive mask pixels:
            #N_R = mask[(mask / obj_attention) >= .5].shape[0]
            #N_B = mask[(mask / obj_attention) < .5].shape[0]
            tmp = (background - original) # np.abs
            juan_mask = ( tmp > 0 ).astype(float) 
            #juan_mask = (np.abs(mask - obj_attention) < .001).astype(float) # mask for the robot
            zero_mask = (~(juan_mask).astype(bool)).astype(float) #((mask - obj_attention) < .5).astype(float) # mask for the background
            #print("robot pixels mask", mask[(mask / obj_attention) > .5].shape)
            #print("back pixels mask", mask[(mask / obj_attention) < .5].shape)
            #print(juan_mask.shape, juan_mask.max(), juan_mask.min())

            #'''
            cond_R = juan_mask[:, :, 0].astype(bool) == True
            cond_B = zero_mask[:, :, 0].astype(bool) == True
            class_masks_R[i][:, :, 0][cond_R] = (cdist(juan_mask * original, juan_mask * reconstructed) < threshold*max_cdist)[cond_R] # RR
            class_masks_R[i][:, :, 1][cond_R] = (cdist(juan_mask * background, juan_mask * reconstructed) < threshold*max_cdist)[cond_R] # RB
            class_masks_R[i][:, :, 1][cond_R] = (~class_masks_R[i][:, :, 0].astype(bool) & class_masks_R[i][:, :, 1].astype(bool))[cond_R]
            class_masks_R[i][:, :, 2][cond_R] = (~class_masks_R[i][:, :, 0].astype(bool) & ~class_masks_R[i][:, :, 1].astype(bool))[cond_R] # RX
            
            class_masks_B[i][:, :, 0][cond_B] = (cdist(zero_mask * original, zero_mask * reconstructed) < threshold*max_cdist)[cond_B] # BB
            class_masks_B[i][:, :, 1][cond_B] = (cdist(zero_mask * background, zero_mask * reconstructed) < threshold*max_cdist)[cond_B] # BR
            class_masks_B[i][:, :, 1][cond_B] = (~class_masks_B[i][:, :, 0].astype(bool) & class_masks_B[i][:, :, 1].astype(bool))[cond_B]
            class_masks_B[i][:, :, 2][cond_B] = (~class_masks_B[i][:, :, 0].astype(bool) & ~class_masks_B[i][:, :, 1].astype(bool))[cond_B] # BX
            
            N_RR = np.sum(class_masks_R[i][:, :, 0][cond_R]) #[class_masks_R[i][:, :, 0] > .5].shape[0]
            N_RB = np.sum(class_masks_R[i][:, :, 1][cond_R]) 
            N_RX = np.sum(class_masks_R[i][:, :, 2][cond_R]) 

            N_BR = np.sum(class_masks_B[i][:, :, 1][cond_B]) #[class_masks_B[i][:, :, 1] > .5].shape[0]
        
            if N_RR + N_RB + N_RX < 0.1:
                IoU = 0
                __R = 0
                __B = 0    
            else:
                IoU = N_RR / (N_RR + N_BR)
                __R = N_RR / (N_RR + N_RB + N_RX)  
                __B = 1.0 / ((N_BR / (N_RR + N_RB + N_RX)) + 1.0) 
            
            minRB = min(__R, __B)
            #print("N_R: ", N_R, "N_RR: ", N_RR, " N_RB: ", N_RB, " N_RX: ", N_RX, " N_BR: ", N_BR)
            
            IoUs[i] = IoU
            minRBs[i] = minRB

            #print(class_masks_B[i].shape, class_masks_B[i].max(), class_masks_B[i].min())
            label = str(np.random.randint(0, 1000))
            
            '''
            imsave("tmp/{}.png".format("{}_original".format(label)), x_test[i])
            imsave("tmp/{}.png".format("{}_decoded".format(label)), decoded_imgs[i])
            imsave("tmp/{}.png".format("{}_Cmask_B".format(label)), class_masks_B[i])
            imsave("tmp/{}.png".format("{}_Cmask_R".format(label)), class_masks_R[i])
            imsave("tmp/{}.png".format("{}_Cmask_combo".format(label)), (class_masks_R[i].astype(int) + class_masks_B[i].astype(int)).astype(int))
            '''        
            #if i > 5:
            #    exit()
        #print("avg. IoU: ", IoUs.mean(), " - ", IoUs.std(), "avg min(R,B): ", minRBs.mean(), " - ", minRBs.std())
        np.savetxt('snapshots/{}.eval'.format(experiment_label), [IoUs.mean(), IoUs.std(), minRBs.mean(), minRBs.std()], delimiter=",", fmt='%1.5f', newline=' ')

        n = 20
        fig = plt.figure(figsize=(200//4, 70//4)) # 20,4 if 10 imgs
        for i in range(n):
            # display original
            ax = plt.subplot(7, n, i+1); plt.yticks([])
            plt.imshow(x_test[i]) #.reshape(img_dim, img_dim)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            
            if i == 0:
                ax.set_ylabel("original", rotation=90, size='xx-large')
                ax.set_yticklabels([])  
            else:
                ax.get_yaxis().set_visible(False)

            # display encoded - vmin and vmax are needed for scaling (otherwise single pixels are drawn as black)
            ax = plt.subplot(7, n, i+1+n); plt.yticks([])
            plt.imshow(encoded_imgs[i].reshape(lat_h, lat_w, lat_ch) if lat_ch == 3 else encoded_imgs[i].reshape(lat_h, lat_w), 
                vmin=encoded_imgs.min(), vmax=encoded_imgs.max())
            plt.gray()
            ax.get_xaxis().set_visible(False)
            #ax.get_yaxis().set_visible(False)

            if i == 0:
                ax.set_ylabel("latent", rotation=90, size='xx-large')
                ax.set_yticklabels([])  
            else:
                ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(7, n, i+1+2*n); plt.yticks([])
            plt.imshow(decoded_imgs[i]) # .reshape(img_dim, img_dim)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            #ax.get_yaxis().set_visible(False)

            if i == 0:
                ax.set_ylabel("decoded", rotation=90, size='xx-large')
                ax.set_yticklabels([])  
            else:
                ax.get_yaxis().set_visible(False)
            
            # display dreamed latent space
            ax = plt.subplot(7, n, i+1+3*n); plt.yticks([])
            plt.imshow(latent_dreams[i].reshape(lat_h, lat_w, lat_ch) if lat_ch == 3 else encoded_imgs[i].reshape(lat_h, lat_w), 
                vmin=encoded_imgs.min(), vmax=encoded_imgs.max())
            plt.gray()
            ax.get_xaxis().set_visible(False)
            #ax.get_yaxis().set_visible(False)

            if i == 0:
                ax.set_ylabel("latent dream", rotation=90, size='xx-large')
                ax.set_yticklabels([])  
            else:
                ax.get_yaxis().set_visible(False)
            
            # display dreamed images 
            ax = plt.subplot(7, n, i+1+4*n); plt.yticks([])
            plt.imshow(dreams[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            #ax.get_yaxis().set_visible(False)

            if i == 0:
                ax.set_ylabel("decoded dream", rotation=90, size='xx-large')
                ax.set_yticklabels([])  
            else:
                ax.get_yaxis().set_visible(False)

            # display masks
            ax = plt.subplot(7, n, i+1+5*n); plt.yticks([])
            plt.imshow(class_masks_R[i]) # .reshape(img_dim, img_dim)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            #ax.get_yaxis().set_visible(False)

            if i == 0:
                ax.set_ylabel("front mask", rotation=90, size='xx-large')
                ax.set_yticklabels([])  
            else:
                ax.get_yaxis().set_visible(False)

            # display masks
            ax = plt.subplot(7, n, i+1+6*n); plt.yticks([])
            plt.imshow(class_masks_B[i]) # .reshape(img_dim, img_dim)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            #ax.get_yaxis().set_visible(False)

            if i == 0:
                ax.set_ylabel("back mask", rotation=90, size='xx-large')
                ax.set_yticklabels([])  
            else:
                ax.get_yaxis().set_visible(False)

        '''
        if interactive:
            plt.show()
        else:
            plt.savefig('snapshots/{}.pdf'.format(experiment_label), bbox_inches='tight')
        '''
        
        fig.savefig('snapshots/{}.pdf'.format(experiment_label), bbox_inches='tight')

        '''
        n = 10
        plt.figure(figsize=(20, 8))
        for i in range(n):
            ax = plt.subplot(1, n, i+1)
            plt.imshow(encoded_imgs[i].reshape(lat_h, lat_w).T)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
        '''