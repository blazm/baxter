import os
#os.environ["TF_USE_CUDNN"]="0"   # set CUDNN to false, since it is non-deterministic: https://github.com/keras-team/keras/issues/2479
#os.environ['PYTHONHASHSEED'] = '0'

# ensure reproducibility
#'''
from numpy.random import seed
seed(3)
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
from models import build_conv_only_ae, make_forward_model
from world_models_vae_arch import build_vae_world_model

from data_generators import data_generator, data_generator_mnist, random_data_generator, brownian_data_generator
from utils import load_parameters, list_images_recursively

from train_ae import prepare_optimizer_object

from numpy.random import seed
seed(6)
from tensorflow import set_random_seed
set_random_seed(6)

if __name__ == "__main__":
    
    from pathlib import Path
    # random generator config
    dir_with_src_images = Path('data/generated_simple/')
    base_image = 'median_image.png'
    object_images = ['circle-red.png', 'robo-green.png'] # circle in the first place, as robo can be drawn over it

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
    include_forward_model = eval(parameters["general"]["include_forward_model"])
    train_only_forward = eval(parameters["general"]["train_only_forward"])

    train_dir = eval(parameters["dataset"]["train_dir"])
    valid_dir = eval(parameters["dataset"]["valid_dir"])
    test_dir = eval(parameters["dataset"]["test_dir"])
    img_shape = eval(parameters["dataset"]["img_shape"])

    total_images = int(parameters["synthetic"]["total_images"])
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

    # # load best models that we have
    '''
    # best AE
    model_label = 'ae_conv'
    latent_size = 16
    conv_layers = 4
    '''
    # best VAE
    model_label = 'vaewm'
    latent_size = 64
    conv_layers = 3
    # '''

    back_attention = 0.2
    obj_attention = 0.8
    size_factor = 3.0

    # vae from world models
    if 'vaewm' in model_label:
        autoencoder, encoder, decoder_mu_log_var, decoder, latent_shape = build_vae_world_model(
            img_shape=img_shape, latent_size=latent_size, 
            opt=opt, loss=loss, #batch_size=batch_size, 
            conv_layers=conv_layers, initial_filters=num_filters) #, kernel_size=kernel_size, kernel_mult=kernel_mult)
    # ae
    elif 'ae_conv' in model_label:
        autoencoder, encoder, decoder, latent_size = build_conv_only_ae(
            img_shape=img_shape, latent_size=latent_size, 
            opt=opt, loss=loss, 
            conv_layers=conv_layers, initial_filters=num_filters) #, kernel_size=kernel_size, kernel_mult=kernel_mult)


    experiment_label = "{}.osize-{}.oatt-{}.e{}.bs{}.lat{}.c{}.opt-{}.loss-{}".format(
        model_label, size_factor, obj_attention, num_epochs, batch_size,
        "{:02d}".format(latent_size) if type(latent_size) == int else 'x'.join(map(str,latent_size[1:])), 
        conv_layers, opt, loss)
    
    print("Preloading weights from previous model: {}".format(experiment_label))
    autoencoder.load_weights('trained_models/{}.h5'.format(experiment_label), by_name=True)

    print("latent size: ", latent_size)
    
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


    forward_model = make_forward_model([latent_size], latent_size)
    forward_model.compile(loss='mse', optimizer=prepare_optimizer_object('adam', 0.001), metrics=['mse'])

    forward_model.summary()

    #if train_dir is None:
    #    x_train, x_test = load_data()
    #    total_images = x_train.shape[0]
    if do_train:
        from keras.callbacks import TensorBoard, CSVLogger, LearningRateScheduler

        fitting_generator = brownian_data_generator(dir_with_src_images, base_image, object_images, img_shape=img_shape, batch_size=batch_size+1)
        valid_generator = brownian_data_generator(dir_with_src_images, base_image, object_images, img_shape=img_shape, batch_size=batch_size+1)

        batches_per_epoch = 100
        num_iterations = num_epochs * batches_per_epoch
        iterations = 0
        history = []
        val_history = []

        from tqdm import tqdm
        with tqdm(total=num_iterations) as pbar:
            pbar.set_description("loss: %f" % -1.0)
            for ([batch_inputs, batch_masks, batch_actions], batch_outputs), val in zip(fitting_generator, valid_generator):
                if iterations >= num_iterations:
                    pbar.close()
                    print("over the iteration limit, stopping")
                    break

                latent = encoder.predict(batch_inputs)
                if len(latent.shape) > 2:
                    bs, h, w, ch = latent.shape
                    latent = np.reshape(latent, (bs, h*w*ch))
                latent_t = latent[0:batch_size]
                latent_t_plus_1 = latent[1:1+batch_size]
                batch_actions = batch_actions[1:1+batch_size]
                #print(batch_actions.shape)
                batch_actions = np.repeat(batch_actions, latent_size, axis=1) # balance the actions with latent_size
                #print(batch_actions.shape)
                # calculate learning rate:
                #if iterations > 0 and iterations % 2000 == 0:
                #    current_learning_rate = K.eval(forward_model.optimizer.lr) / 10.
                #    K.set_value(forward_model.optimizer.lr, current_learning_rate)  # set new lr

                result = forward_model.train_on_batch([latent_t, batch_actions], latent_t_plus_1)
                
                ([batch_inputs, batch_masks, batch_actions], batch_outputs) = val
                latent = encoder.predict(batch_inputs)
                if len(latent.shape) > 2:
                    bs, h, w, ch = latent.shape
                    latent = np.reshape(latent, (bs, h*w*ch))
                latent_t = latent[0:batch_size]
                latent_t_plus_1 = latent[1:1+batch_size]
                batch_actions = batch_actions[1:1+batch_size]

                batch_actions = np.repeat(batch_actions, latent_size, axis=1)

                val_result = forward_model.test_on_batch([latent_t, batch_actions], latent_t_plus_1)
                
                history.append(result[0])
                val_history.append(val_result[0])
                
                #print(iterations)

                iterations += 1
                pbar.set_description("loss: %f, val_loss: %f, lr: %f" % (result[0], val_result[0], K.eval(forward_model.optimizer.lr)))
                pbar.update(1)

                
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(history, color='red')
        plt.plot(val_history, color='blue')
        #plt.show()
        fig.savefig('snapshots/forward_loss_{}.png'.format(experiment_label), bbox_inches='tight')

        forward_model.save('trained_models/forward_model_{}.h5'.format(experiment_label))

    if do_test:
        forward_model.load_weights('trained_models/forward_model_{}.h5'.format(experiment_label))

        # test the trained forward model:
        valid_generator = brownian_data_generator(dir_with_src_images, base_image, object_images, img_shape=img_shape, batch_size=batch_size+1)
        
        h,w,ch = img_shape
        x_test = np.zeros((batch_size+1, h, w, ch))
        x_mask = np.zeros((batch_size+1, h, w, ch))
        x_action = np.zeros((batch_size+1, 1))
       

        [x_test, x_mask, x_action], out = valid_generator.__next__()
        x_test = x_test[0:batch_size]
        x_action = x_action[1:1+batch_size]
        x_action = np.repeat(x_action, latent_size, axis=1)
        print(x_action[1])
        '''
        i = 0
        for [img, mask, action], out in valid_generator:
            
            x_test[i] = img[0] #resize(img, (h, w), anti_aliasing=True)
            #print(mask[0].shape, mask[0].min(), mask[0].max())
            x_mask[i] = mask[0]
            x_action[i] = action[0]
            i += 1
            if i >= batch_size+1:
                break
        '''

        back_generator = brownian_data_generator(dir_with_src_images, base_image, object_images, img_shape=img_shape, batch_size=batch_size+1)
        [batch_inputs, batch_masks, batch_actions], batch_outputs = next(back_generator)
        background = np.median(batch_outputs, axis=0, keepdims=False)
        
        #decoded_imgs = autoencoder.predict(x_test)
        print("input images shape: ", x_test.shape)
        encoded_imgs = encoder.predict(x_test)

        if len(encoded_imgs.shape) > 2:
            bs, h, w, ch = encoded_imgs.shape
            tmp = np.reshape(encoded_imgs, (bs, h*w*ch))
            encoded_imgs_t_plus_1 = forward_model.predict([tmp, x_action])
        else:
            encoded_imgs_t_plus_1 = forward_model.predict([encoded_imgs, x_action])
        if type(encoded_imgs_t_plus_1) == list:
            encoded_imgs_t_plus_1 = encoded_imgs_t_plus_1[-1]

        print("encoded images shape: ", encoded_imgs.shape) #, encoded_imgs[1].shape, encoded_imgs[2].shape)
        print("encoded MAX / MIN: ", encoded_imgs.max(), " / ", encoded_imgs.min())
        
        # normalize before displaying
        #encoded_imgs = (encoded_imgs - encoded_imgs.min()) / np.ptp(encoded_imgs) * 255.0
        #print(encoded_imgs)
        decoded_imgs = decoder.predict(encoded_imgs)
        if len(encoded_imgs.shape) > 2:
            bs, h, w, ch = encoded_imgs.shape
            tmp = np.reshape(encoded_imgs_t_plus_1, (bs, h, w, ch))
            decoded_imgs_t_plus_1 = decoder.predict(tmp)
        else:
            decoded_imgs_t_plus_1 = decoder.predict(encoded_imgs_t_plus_1)
        print(decoded_imgs.shape)
        print(decoded_imgs_t_plus_1.shape)
        print("h, w, ch: {},{},{}".format(lat_h, lat_w, lat_ch))
        print("encoded MAX / MIN: ", encoded_imgs.max(), " / ", encoded_imgs.min())
        
        print("input images shape: ", x_test.shape)
        print("decoded_imgs images shape: ", decoded_imgs.shape)
        #latent_dreams = encoder.predict(decoded_imgs)
        #if include_forward_model:
        #latent_dreams = forward_model.predict([latent_dreams, x_action])
        #dreams = decoder.predict(latent_dreams) # dream the images

        print("decoded images shape: ", decoded_imgs.shape)
        import os
        import matplotlib as mpl
        if os.environ.get('DISPLAY','') == '':
            print('No display found. Using non-interactive Agg backend')
            mpl.use('Agg')

        import matplotlib.pyplot as plt

        n = batch_size
        fig = plt.figure(figsize=(int(n * 2.5), int(n * 0.5))) # 20,4 if 10 imgs
        for i in range(n):
            # display original
            ax = plt.subplot(6, n, i+1); plt.yticks([])
            plt.imshow(x_test[i]) #.reshape(img_dim, img_dim)
            plt.gray()
            ax.get_xaxis().set_visible(True)
            ax.set_xlabel("dx:{:1.2f} dy:{:1.2f}".format(x_action[i][0], x_action[i][-1]), rotation=0, size='x-large')
            ax.set_xticklabels([])  
            
            if i == 0:
                ax.set_ylabel("original t", rotation=90, size='xx-large')
                ax.set_yticklabels([])  
            else:
                ax.get_yaxis().set_visible(False)

            # display encoded - vmin and vmax are needed for scaling (otherwise single pixels are drawn as black)
            ax = plt.subplot(6, n, i+1+n); plt.yticks([])
            plt.imshow(encoded_imgs[i].reshape(lat_h, lat_w, lat_ch) if lat_ch == 3 else encoded_imgs[i].reshape(lat_h, lat_w), 
                vmin=encoded_imgs.min(), vmax=encoded_imgs.max(), interpolation='nearest')
            plt.gray()
            ax.get_xaxis().set_visible(False)
            #ax.get_yaxis().set_visible(False)

            if i == 0:
                ax.set_ylabel("encoded t", rotation=90, size='xx-large')
                ax.set_yticklabels([])  
            else:
                ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(6, n, i+1+2*n); plt.yticks([])
            plt.imshow(decoded_imgs[i], vmin=decoded_imgs[i].min(), vmax=decoded_imgs[i].max()) # .reshape(img_dim, img_dim)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            #ax.get_yaxis().set_visible(False)

            if i == 0:
                ax.set_ylabel("decoded t", rotation=90, size='xx-large')
                ax.set_yticklabels([])  
            else:
                ax.get_yaxis().set_visible(False)
            
            
            # display masks
            ax = plt.subplot(6, n, i+1+3*n); plt.yticks([])
            plt.imshow(encoded_imgs_t_plus_1[i].reshape(lat_h, lat_w, lat_ch) if lat_ch == 3 else encoded_imgs_t_plus_1[i].reshape(lat_h, lat_w), 
                vmin=encoded_imgs.min(), vmax=encoded_imgs.max(), interpolation='nearest')
            plt.gray()
            ax.get_xaxis().set_visible(False)
            #ax.get_yaxis().set_visible(False)

            if i == 0:
                ax.set_ylabel("predicted t+1", rotation=90, size='xx-large')
                ax.set_yticklabels([])  
            else:
                ax.get_yaxis().set_visible(False)

            # display dreamed latent space
            ax = plt.subplot(6, n, i+1+4*n); plt.yticks([])
            plt.imshow(decoded_imgs_t_plus_1[i], vmin=decoded_imgs.min(), vmax=decoded_imgs.max(), interpolation='nearest')
            plt.gray()
            ax.get_xaxis().set_visible(False)
            #ax.get_yaxis().set_visible(False)

            if i == 0:
                ax.set_ylabel("decoded t+1", rotation=90, size='xx-large')
                ax.set_yticklabels([])  
            else:
                ax.get_yaxis().set_visible(False)
            
            # display dreamed images 
            try:
                ax = plt.subplot(6, n, i+1+5*n); plt.yticks([])
                plt.imshow(x_test[i+1], vmin=x_test[i+1].min(), vmax=x_test[i+1].max())
                plt.gray()
            except:
                #plt.imshow(np.ones((2,2))*255, vmin=0, vmax=255)
                plt.gray()

            ax.get_xaxis().set_visible(False)
            #ax.get_yaxis().set_visible(False)

            if i == 0:
                ax.set_ylabel("original t+1", rotation=90, size='xx-large')
                ax.set_yticklabels([])  
            else:
                ax.get_yaxis().set_visible(False)
        
        fig.savefig('snapshots/forward_{}.pdf'.format(experiment_label), bbox_inches='tight')
