# coding=utf-8
from os import system
from utils import set_parameter

if __name__ == "__main__":
    parameters_filepath = "config.ini"

    do_train = True
    do_test = False

    # all possible hyperparameter configurations
    #latent_dims = [4, 9, 16, 25, 36, 49, 64, 81, 100] #  , # more or less the same for the rest of the models
    #optimizers = ['adam'] # , 'adadelta'
    #losses = ['mse', 'wmse'] # , 'bin-xent'
    #models = ['ae_conv', 'ae_dense', 'vae']
    #generator_modes = ["sub-mean-noisy", "sub-mean", "noisy", "default"]

    # TODO: run experiments
    latent_dims = [4, 16, 64, 256] 
    # latent sizes map directly to number of conv layers for AE_conv 
    conv_layers = [5, 4, 3, 2]
    #obj_weights = [0.1, 0.2, 0.5, 0.8, 0.9, 1.0] # back_weight = 1.0 - obj_weight, loss=wmse, opt=adamw
    obj_weights = [0.33, 0.5, 0.67, 0.8, 0.94, 0.98] # as suggested by Abraham
    models = ['ae_conv', 'vaewm']
    obj_sizes = [1.0, 2.0, 3.0, 4.0]
    # 64x64x3 - VAE is fixed to this image size
    

    if do_train:
        '''
        for gen_m in generator_modes:
            for opt in optimizers:
                for loss in losses:
        '''
        for size_factor in obj_sizes:
            for obj_weight in obj_weights:
                for model in models:    
                    for lat_dim, conv_lyrs in zip(latent_dims, conv_layers):
              
                        set_parameter(parameters_filepath, "general", "do_train", "True")
                        set_parameter(parameters_filepath, "general", "do_test", "True")

                        set_parameter(parameters_filepath, "synthetic", "size_factor", str(size_factor))
                        set_parameter(parameters_filepath, "synthetic", "obj_attention", str(obj_weight))
                        set_parameter(parameters_filepath, "synthetic", "back_attention", str("%.2f" % round(1.0 - obj_weight, 2))

                        set_parameter(parameters_filepath, "hyperparam", "latent_size", str(lat_dim))
                        set_parameter(parameters_filepath, "hyperparam", "conv_layers", str(conv_lyrs))
                        
                        set_parameter(parameters_filepath, "hyperparam", "model_label", model)
                        #set_parameter(parameters_filepath, "hyperparam", "loss", loss)
                        #set_parameter(parameters_filepath, "hyperparam", "opt", opt)
                        
                        # run training for this exact configuration
                        # DONE: save parameter configuration to enable experiment reproduction (in train.py)
                        system("python train_ae.py") # TODO: change this to be compatible with Linux as well -- system("source activate tf35;python train_ae.py")
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