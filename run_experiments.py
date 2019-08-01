# coding=utf-8
from os import system
from utils import set_parameter

import subprocess
from tqdm import trange

import platform

if __name__ == "__main__":
    parameters_filepath = "config.ini"

    do_train = True
    do_test = True

    # all possible hyperparameter configurations
    #latent_dims = [4, 9, 16, 25, 36, 49, 64, 81, 100] #  , # more or less the same for the rest of the models
    #optimizers = ['adam'] # , 'adadelta'
    #losses = ['mse', 'wmse'] # , 'bin-xent'
    #models = ['ae_conv', 'ae_dense', 'vae']
    #generator_modes = ["sub-mean-noisy", "sub-mean", "noisy", "default"]

    # TODO: run experiments
    latent_dims = [16, 64]  # [4, 
    # latent sizes map directly to number of conv layers for AE_conv 
    conv_layers = [4, 3] # [5, 
    #obj_weights = [0.1, 0.2, 0.5, 0.8, 0.9, 1.0] # back_weight = 1.0 - obj_weight, loss=wmse, opt=adamw
    obj_weights = [0.33, 0.67, 0.8, 0.98] # as suggested by Abraham # 0.94, 0.5, 
    models = ['ae_conv', 'vaewm']
    obj_sizes = [1.0, 2.0, 3.0] #, 4.0]
    # 64x64x3 - VAE is fixed to this image size
    
    obj_weights.reverse()
    obj_sizes.reverse()

    if do_train:
        '''
        for gen_m in generator_modes:
            for opt in optimizers:
                for loss in losses:
        '''
        for s in trange(len(obj_sizes)):
            for w in trange(len(obj_weights)):
                for m in trange(len(models)):    
                    for l in trange(len(latent_dims)):
                        
                        size_factor = obj_sizes[s]
                        obj_weight = obj_weights[w]
                        model = models[m]
                        lat_dim = latent_dims[l]
                        conv_lyrs = conv_layers[l]
                        
                        set_parameter(parameters_filepath, "general", "do_train", "True")
                        set_parameter(parameters_filepath, "general", "do_test", "True")

                        set_parameter(parameters_filepath, "synthetic", "size_factor", str(size_factor))
                        set_parameter(parameters_filepath, "synthetic", "obj_attention", str(obj_weight))
                        set_parameter(parameters_filepath, "synthetic", "back_attention", str("%.2f" % round(1.0 - obj_weight, 2)))

                        set_parameter(parameters_filepath, "hyperparam", "latent_size", str(lat_dim))
                        set_parameter(parameters_filepath, "hyperparam", "conv_layers", str(conv_lyrs))
                        
                        set_parameter(parameters_filepath, "hyperparam", "model_label", model)
                        #set_parameter(parameters_filepath, "hyperparam", "loss", loss)
                        #set_parameter(parameters_filepath, "hyperparam", "opt", opt)
                        
                        # run training for this exact configuration
                        # DONE: save parameter configuration to enable experiment reproduction (in train.py)
                        #system("python train_ae.py") # TODO: change this to be compatible with Linux as well -- system("source activate tf35;python train_ae.py")
                        if platform.system() == 'Windows':
                            subprocess.check_output('python train_ae.py', shell=True) 
                        elif platform.system() == 'Linux':
                            system("source activate python35;python train_ae.py")
                        else:
                            print("OS not recognized. Exiting.")
                            exit()
                        # results will be saved as: trained models, tensorboard logs, pdf, csv, ini

    # once trained, we can test all the models
    '''
    if do_test:
        for gen_m in generator_modes:
            for opt in optimizers:
                for loss in losses:
                    for lat_dim in latent_dims:
                        set_parameter(parameters_filepath, "general", "do_train", "False")
                        set_parameter(parameters_filepath, "general", "do_test", "True")

                        set_parameter(parameters_filepath, "dataset", "generator_mode", gen_m)

                        set_parameter(parameters_filepath, "hyperparam", "latent_size", str(lat_dim))
                        set_parameter(parameters_filepath, "hyperparam", "loss", loss)
                        set_parameter(parameters_filepath, "hyperparam", "opt", opt)
                        # run training for this exact configuration
                        system("python train_ae.py") # TODO: change this to be compatible with Linux as well -- system("source activate tf35;python train_ae.py")
                        # results will be saved as: trained models, tensorboard logs
    
    '''
