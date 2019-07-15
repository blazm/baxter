"""
Python module for scripting helper functions
"""
from glob import glob
import configparser
import os
import re
import json

def load_parameters(parameters_filepath):
    """
    load parameters from an *.ini file
    :param parameters_filepath: filename (absolute path)
    :return: nested dictionary of parameters
    """
    conf_parameters = configparser.ConfigParser()
    conf_parameters.read(parameters_filepath, encoding="UTF-8")
    # nested_parameters = utils.convert_configparser_to_dictionary(conf_parameters)
    nested_parameters = {s: dict(conf_parameters.items(s)) for s in conf_parameters.sections()}
    return nested_parameters

def set_parameter(parameters_filepath, section_name, parameter_name, parameter_value):
    """
    set the specified parameter to the specified value and write back to the *.ini file
    :param parameters_filepath: filename (absolute path)
    :param section_name: section name under which parameter is
    :param parameter_name: parameter name
    :param parameter_value: target value
    :return:
    """
    conf_parameters = configparser.ConfigParser()
    conf_parameters.read(parameters_filepath, encoding="UTF-8")
    conf_parameters.set(section_name, parameter_name, parameter_value)
    with open(parameters_filepath, 'w') as config_file:
        conf_parameters.write(config_file)


# PSNR metric
import numpy as np
from numpy import float32,  uint32

from scipy.misc import imresize, imsave #, imread, imsave
from scipy.ndimage import imread

import tensorflow as tf
import keras.backend as K

def log10(x):
    """
    there is not direct implementation of log10 in TF.
    But we can create it with the power of calculus.
    Args:
        x (array): input array

    Returns: log10 of x

    """
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def psnr(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10. * log10(K.mean(K.square(y_pred - y_true)))


# other helper functions

def read_files(path, isDir=False):
    '''Read filenames in the path. '''
    filelist = []
    for item in os.listdir(path):
       # ident, sess, num = item.split('_')
       # ids.append(int(ident))
        if isDir and os.path.isdir(os.path.join(path, item)):
            filelist.append(item)
        elif not isDir and not os.path.isdir(os.path.join(path, item)):
            filelist.append(item)
        #os.path.join(path,item)
    return filelist

def list_images_recursively(path, isDir=False):
    '''Grab any image filepaths in the path. '''
    old = os.getcwd()
    new = os.path.join(old, path)
    #print(old, new)
    os.chdir(new)

    file_set = set()

    for root, dirs, files in os.walk("."):
        for f in files:
            if '.png' in f or '.jpg' in f or '.jpeg' in f:
                file_set.add(os.path.join(root, f))

    os.chdir(old)
    return list(file_set)

def trim(image, image_size, trim=24, top=24):
    
    #height, width = None, None
    #if len(image.shape) == 3:
    height, width, d = image.shape
    #else:
    #    height, width = image.shape
    #    d = 1

    width = int(width-2*trim)
    height = int(width*image_size[0]/image_size[1])

    #if len(image.shape) == 3:
    image = image[trim+top:trim+height,trim:trim+width,:]
    #else: 
    #    image = image[trim+top:trim+height,trim:trim+width,:]
    # Resize and fit between 0-1
    image = imresize( image, image_size )
    image = image / 255.0

    return image

def preprocess_size(image, deconv_layers=5, initial_shape=(5,4)):
    #alt_img = src_images[0][:,:,:3]
    #print("ALT IMG shape: ", alt_img.shape)
    # crop face
    #initial_shape = (5, 4)
    #deconv_layers = 5
    h, w = initial_shape
    new_scale = 2**(deconv_layers+1)
    new_dim = (h*new_scale, w*new_scale)  # (600, 480)
    image = trim(image, new_dim, trim=130, top=48)

    return image

from scipy import ndimage
def preprocess_enhance_edges(image):

    blurred_f = np.empty(image.shape)
    filter_blurred_f = np.empty(image.shape)

    blurred_f[:,:,0] =  ndimage.gaussian_filter(image[:,:,0], 3)
    blurred_f[:,:,1] =  ndimage.gaussian_filter(image[:,:,1], 3)
    blurred_f[:,:,2] =  ndimage.gaussian_filter(image[:,:,2], 3)

    #blurred_f = ndimage.gaussian_filter(image, 3)
    #increase the weight of edges by adding an approximation of the Laplacian:

    filter_blurred_f[:,:,0] =  ndimage.gaussian_filter(blurred_f[:,:,0], 3)
    filter_blurred_f[:,:,1] =  ndimage.gaussian_filter(blurred_f[:,:,1], 3)
    filter_blurred_f[:,:,2] =  ndimage.gaussian_filter(blurred_f[:,:,2], 3)


    #filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
    alpha = 0.3
    sharpened = (1.0 - alpha) * blurred_f + alpha * (blurred_f - filter_blurred_f)
    return sharpened
    #edges = filters.scharr(image)


def loadAndResizeImages2(path, names=[], preprocessors=[], load_alpha=False):
    images = []
    if not names:
        names = read_files(path)

    for name in names:
        if '.png' in name and load_alpha:
            img = imread(os.path.join(path, name), mode='RGBA')
        else:
            img = imread(os.path.join(path, name), mode='RGB')
        #img = preprocess_enhance_edges(img)
        #img = preprocess_size(img, deconv_layers=deconv_layers)
        images.append(img)

    for p in preprocessors:
        images = map(p, images)
    images = list(images) # convert from map to list

    np_images = np.array(images)

    return np_images


def preprocess_size(image, new_dim=(5,4)):
    image = trim(image, new_dim, trim=0, top=0)
    return image