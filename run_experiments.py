# coding=utf-8
from os import system
from utils import set_parameter

if __name__ == "__main__":
    parameters_filepath = "config.ini"

    do_train = True
    do_test = False

    # all possible hyperparameter configurations
    latent_dims = [4, 9, 16, 25, 36, 49, 64, 81, 100] #  , # more or less the same for the rest of the models
    optimizers = ['adam'] # , 'adadelta'
    losses = ['mse', 'wmse'] # , 'bin-xent'
    models = ['ae_conv', 'ae_dense', 'vae']
    generator_modes = ["sub-mean-noisy", "sub-mean", "noisy", "default"]

    if do_train:
        for gen_m in generator_modes:
            for opt in optimizers:
                for loss in losses:
                    for lat_dim in latent_dims:
                        set_parameter(parameters_filepath, "general", "do_train", "True")
                        set_parameter(parameters_filepath, "general", "do_test", "True")

                        set_parameter(parameters_filepath, "dataset", "generator_mode", gen_m)

                        set_parameter(parameters_filepath, "hyperparam", "latent_size", str(lat_dim))
                        set_parameter(parameters_filepath, "hyperparam", "loss", loss)
                        set_parameter(parameters_filepath, "hyperparam", "opt", opt)
                        # run training for this exact configuration
                        
                        # TODO: save parameter configuration to enable experiment reproduction
                        system("python train_ae.py") # TODO: change this to be compatible with Linux as well -- system("source activate tf35;python train_ae.py")
                        # results will be saved as: trained models, tensorboard logs

    # once trained, we can test all the models
    
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
    '''