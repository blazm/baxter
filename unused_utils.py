import numpy as np
from numpy import float32,  uint32

from scipy.misc import imresize, imsave #, imread, imsave
from scipy.ndimage import imread

import tensorflow as tf
import keras.backend as K

def resize_roi_05x(roi):
    '''VGG Extractor works better if the whole face image is passed to the net, so here we resize roi by factor of 2'''
    x, y, w, h = roi
    
    w = int(w/2)
    h = int(h/2)

    x = int(x + w/2.0)
    y = int(y + h/2.0)
    
    return (x, y, w, h)

def resize_roi_2x(roi):
    '''VGG Extractor works better if the whole face image is passed to the net, so here we resize roi by factor of 2'''
    x, y, w, h = roi
    x = int(x - w/2)
    y = int(y - h/2)
    w = 2*w
    h = 2*h
    
    return (x, y, w, h)
    
def crop_from(img, roi):
    x, y, w, h = roi     
    ih, iw, ch = img.shape
    
    dx = 0
    if (x < 0):
        dx = abs(x)
        x = 0
        w = w - dx
    elif (x+w > iw):
        w = iw - x
        
    dy = 0
    if (y < 0):
        dy = abs(y)
        y = 0
        h = h - dy
    elif (y+h > ih):
        h = ih - y
    
    return img[y:y+h, x:x+w, :]
   
def normalize(image, rescale_to=255):
    return (image - np.min(image)) / (np.max(image) - np.min(image)) *rescale_to


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

def read_lines(filename):
    with open(filename, "r") as f: 
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    return lines

#def prepare_descriptions(lines):
#    print("Raw descriptions: ", lines)
#    pass # TODO

def loadImages(path, names=[], newdim=None):
    images = []
    if not names:
        names = read_files(path)

    for name in names:
        img = imread(os.path.join(path, name), mode='RGBA')
        if newdim:
            h, w, ch = img.shape
            img = imresize(img, (newdim, newdim, ch), interp='bicubic')
        images.append(img)
    images = np.array(images)
    #print images.shape

    return images

def loadAndResizeImages(path, names=[], deconv_layers=5, include_neutral=False):
    images = []
    if not names:
        names = read_files(path)

    if include_neutral:
        names = [n for n in names if 'neutral' in n]

    for name in names:
        img = imread(os.path.join(path, name), mode='RGB')
        img = preprocess_enhance_edges(img)
        img = preprocess_size(img, deconv_layers=deconv_layers)
        images.append(img)
    images = np.array(images)
    #print images.shape

    return images

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

def shape_to_nparray(shape,  displacement=(0, 0)):
    """
    Reshapes Shape from dlib predictor to numpy array
    Args:
        shape (dlib Shape object): input shape points
        displacement (x, y): tuple with displacement components, which are subtracted

    Returns: numpy array consisting of shape points
    """
    dx, dy = displacement
    np_arr = []
    for i in range(0,  shape.num_parts):
        np_arr.append((shape.part(i).x - dx,  shape.part(i).y - dy))
    return np.array(np_arr)

def get_landmarks(image, predictor):

    h, w, ch = image.shape
    ix = uint32(0).item()
    iy = uint32(0).item()
    iw = uint32(w).item()
    ih = uint32(h).item()

    src_detection = dlib.rectangle(ix, iy, ix+iw, iy+ih)
    src_shape = predictor(image, src_detection)
    src_pts = shape_to_nparray(src_shape)
    return src_pts

def DEBUG_add_landmarks(image, land_pts, ch=1):
    for x, y in land_pts:
        image[y-1:y+1, x-1:x+1, ch] = 255

def deconv2shape(deconv_layers, initial_shape=(5,4)):
    h, w = initial_shape
    new_scale = 2**(deconv_layers+1)
    return (h*new_scale, w*new_scale) 

def shape2deconv(shape, initial_shape=(5,4)):
    from math import log

    h, w = initial_shape
    hs, ws = shape
    scale = float(hs) / float(h)
    deconv_layers = int(log(scale, 2))-1

    print("deconv layers: ", deconv_layers)
    return deconv_layers


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


def preprocess_augmentations(alt_img, predictor, accessories, enabled, __debug=False):
    # TODO: add augmentations to images (glasses, facials, beards)
    new_dim = alt_img.shape

    copied_im = np.copy(alt_img)

    acc_images, acc_files = accessories

    src_pts = get_landmarks(alt_img, predictor)
    #print("landmarks: ", src_pts)

    # index 39 and 42 represent 40 and 43 landmark points (middle eye points)
    glasses_x, glasses_y = (src_pts[39]+src_pts[42])//2 
    # index 33 and 51 represent 34 and 50 landmark points (bottom of nose and top of lips)
    moustache_x, moustache_y = (src_pts[33]+src_pts[66])//2 
    
    if __debug:
        print("glasses pos: ", glasses_x, glasses_y)
        print("moustache pos: ", moustache_x, moustache_y)

    # DEBUG: show landmarks
    #alt_img[glasses_y-5:glasses_y+5, glasses_x-5:glasses_x+5, 2] = 255
    #DEBUG_add_landmarks(alt_img, src_pts, ch=1)
    
    #dest_path = './before_demo.png'
    #imsave(dest_path, alt_img)

    for gen_img, filename, enable in zip(acc_images, acc_files, enabled):

        if not enable:
            continue

        if "glasses" in filename:
            target_width = int(new_dim[1]*0.7) # slightly smaller glasses than the width of an image (70%)
        elif "moustache" in filename:
            target_width = int(new_dim[1]*0.5) # beard / moustaches of size 50% of the width

        h, w, ch = gen_img.shape
        if __debug:
            print("NEW H: ", int((target_width*h)/w))
        gen_img = imresize(gen_img, (int((target_width*h)/w), target_width, 1), interp='bicubic')

        if __debug:
            print("gen shape: ", gen_img.shape)
            print("alt image: ", alt_img.shape)


        h, w, ch = gen_img.shape
        
        if "glasses" in filename:
            x = (int)(glasses_x) - w//2
            y = (int)(glasses_y) - h//2
        elif "moustache" in filename:
            x = (int)(moustache_x) - w//2
            y = (int)(moustache_y) - h//2

        if __debug:
            print("acc pos: ", x,y)

        mask = gen_img[:, :, 3] / 255.0 # last channel (alpha) is the mask (normalize first!)

        if __debug:
            print("mask shape: ", mask.shape)

        # merge
        try:
            copied_im[y:y+h, x:x+w, 0] = alt_img[y:y+h, x:x+w, 0] * (1-mask) + mask * gen_img[:, :, 0] #*255.0
            copied_im[y:y+h, x:x+w, 1] = alt_img[y:y+h, x:x+w, 1] * (1-mask) + mask * gen_img[:, :, 1] #*255.0
            copied_im[y:y+h, x:x+w, 2] = alt_img[y:y+h, x:x+w, 2] * (1-mask) + mask * gen_img[:, :, 2] #*255.0
            #alt_img[:,:,:,3] = 1;
            #alt_img[y:y+h, x:x+w, 0] = alt_img[y:y+h, x:x+w, 0] * (1-mask) + mask * gen_img[:, :, 0] #*255.0
            #alt_img[y:y+h, x:x+w, 1] = alt_img[y:y+h, x:x+w, 1] * (1-mask) + mask * gen_img[:, :, 1]  #*255.0
            #alt_img[y:y+h, x:x+w, 2] = alt_img[y:y+h, x:x+w, 2] * (1-mask) + mask * gen_img[:, :, 2]  #*255.0
        except:
            import sys
            print("Fit Generator: ERROR: replace - unexpected error:", sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2])
            #print("ERROR: {}, {}, {}".format(gen_img[:, :, 0].shape, alt_img[y:y+h, x:x+w, 0].shape, mask.shape))
            raise

    return copied_im

import pprint
pp = pprint.PrettyPrinter(indent=4)


def data_generator(inputs, outputs, accessories, batch_size, landmark_predictor):
    # Create empty arrays to contain batch of inputs and outputs#
    import random 
    #dummy_input = np.zeros(inputs[0].shape)
    dummy_output = np.zeros(outputs[0].shape)
    h, w, ch = dummy_output.shape

    #print(outputs.shape)
    #print("IN SHAPE: ", inputs[1]['skin'].shape)
    batch_inputs = {}
    for key in inputs[1].keys():
        batch_inputs[key] = np.zeros((batch_size, inputs[1][key].shape[1]))

    #print(batch_inputs)
    #batch_inputs = {
    #        'emotion'    : np.empty((batch_size, len(Emotion.neutral))),
    #        'identity'   : np.empty((batch_size, self.num_identities)),
    #        'orientation': np.empty((batch_size, 2)),
    #    }

    #batch_inputs = [{}] * batch_size # this would be dictionaries
    batch_outputs = np.zeros((batch_size, h, w, ch))

    while True:

        #batch_ix=0
        for i in range(batch_size):
            # choose random index in inputs
            index = random.choice(list(inputs.keys())) # identity index
            array_index = list(inputs.keys()).index(index) # index of identity index in array

            # TODO: define if glasses will be added:  
            add_glasses = np.random.uniform(0.0, 1.0, 1)[0] < 0.5

        
            for key in inputs[index].keys():
                #print(key, inputs[index][key])
                batch_inputs[key][i] = inputs[index][key][:]
            
            add_moustache = False
            if batch_inputs['gender'][i][0] == 1: # if male
                add_moustache = True # np.random.uniform(0.0, 1.0, 1)[0] < 0.75 


            batch_inputs['identity'][i][array_index] = 1.0
            batch_inputs['noise'][i] = np.random.uniform(0.0, 1.0, 25)
            

            batch_outputs[i] = preprocess_augmentations(outputs[array_index], landmark_predictor, accessories, [add_glasses, add_moustache])


            if add_glasses:
                batch_inputs['misc'][i][0] = 1.0 # glasses flag
            else:
                batch_inputs['misc'][i][0] = 0.0 # glasses flag
                batch_outputs[i] = outputs[array_index]

            if add_moustache:
                batch_inputs['misc'][i][1] = 1.0 # moustache flag
            #    batch_outputs[i] = preprocess_augmentations(batch_outputs[i], landmark_predictor, ([acc_images[1]], [acc_files[1]]))
            #    pass
            else:
                batch_inputs['misc'][i][1] = 0.0 # moustache flag
            #    batch_outputs[i] = outputs[array_index]

            #imsave(os.path.join('./tmp/', 'batch{}index{}glasses{}.png'.format(batch_ix,i,add_glasses)), batch_outputs[i])
            #batch_ix += 1

            batch_outputs[i] = batch_outputs[i] / 255.0
        yield batch_inputs, batch_outputs

def preprocess_input(input_dicts):
    print("Input preprocess_input: ", input_dicts)
    return input_dicts

def preprocess_output(output_image, landmark_predictor, accessories):
    #output_images[ix] = preprocess_size(img, deconv_layers)
    output_image = preprocess_augmentations(output_image, landmark_predictor, accessories)

    # update input labels:
    #input_dicts[ix] = 

    # return information about added accessories, so that they can be added to labels
    return output_image


if __name__ == '__main__':
    
    src_descriptions = "../rafd_frontal_descriptions.txt"
    src_path = "../de-id/DB/rafd2-frontal/"
    src_files = read_files(src_path)
    src_images = loadImages(src_path, [src_files[25]])


    acc_path = "./accessories/"
    acc_files = sorted(read_files(acc_path))

    acc_glasses = [f for f in acc_files if "glasses" in f]
    acc_facials = [f for f in acc_files if "moustache" in f]
    acc_beards = [f for f in acc_files if "beard" in f]

    print(acc_glasses)
    exit()
    acc_files = acc_glasses # [acc_glasses[4], acc_facials[2]] # pick subset
    acc_images = loadImages(acc_path, acc_files)


    alt_img = src_images[0][:,:,:3]
    alt_img = preprocess_size(alt_img, deconv_layers=6)
    print("ALT IMG shape: ", alt_img.shape)
    new_dim = alt_img.shape

    # detect landmarks here
    predictor_path = "../de-id/replacer/shape_predictor_68_face_landmarks.dat"
    #detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    src_pts = get_landmarks(alt_img, predictor)
    #print("landmarks: ", src_pts)

    # index 39 and 42 represent 40 and 43 landmark points (middle eye points)
    glasses_x, glasses_y = (src_pts[39]+src_pts[42])//2 
    # index 33 and 51 represent 34 and 50 landmark points (bottom of nose and top of lips)
    moustache_x, moustache_y = (src_pts[33]+src_pts[51])//2 
    
    print("glasses pos: ", glasses_x, glasses_y)
    print("moustache pos: ", moustache_x, moustache_y)

    # DEBUG: show landmarks
    #alt_img[glasses_y-5:glasses_y+5, glasses_x-5:glasses_x+5, 2] = 255
    DEBUG_add_landmarks(alt_img, src_pts, ch=1)
    
    dest_path = './before_demo.png'
    imsave(dest_path, alt_img)

    for gen_img, filename in zip(acc_images, acc_files):

        if "glasses" in filename:
            target_width = int(new_dim[1]*0.7) # slightly smaller glasses than the width of an image (70%)
        elif "moustache" in filename:
            target_width = int(new_dim[1]*0.5) # beard / moustaches of size 50% of the width

        h, w, ch = gen_img.shape
        print("NEW H: ", int((target_width*h)/w))
        gen_img = imresize(gen_img, (int((target_width*h)/w), target_width, 1), interp='bicubic')

        print("gen shape: ", gen_img.shape)
        print("alt image: ", alt_img.shape)

        h, w, ch = gen_img.shape
        
        if "glasses" in filename:
            x = (int)(glasses_x) - w//2
            y = (int)(glasses_y) - h//2
        elif "moustache" in filename:
            x = (int)(moustache_x) - w//2
            y = (int)(moustache_y) - h//2

        print("acc pos: ", x,y)

        mask = gen_img[:, :, 3] / 255.0 # last channel (alpha) is the mask (normalize first!)

        print("mask shape: ", mask.shape)

        # merge
        try:
            alt_img[y:y+h, x:x+w, 0] = alt_img[y:y+h, x:x+w, 0] * (1-mask) + mask * gen_img[:, :, 0] #*255.0
            alt_img[y:y+h, x:x+w, 1] = alt_img[y:y+h, x:x+w, 1] * (1-mask) + mask * gen_img[:, :, 1] #*255.0
            alt_img[y:y+h, x:x+w, 2] = alt_img[y:y+h, x:x+w, 2] * (1-mask) + mask * gen_img[:, :, 2] #*255.0
            #alt_img[:,:,:,3] = 1;
            #alt_img[y:y+h, x:x+w, 0] = alt_img[y:y+h, x:x+w, 0] * (1-mask) + mask * gen_img[:, :, 0] #*255.0
            #alt_img[y:y+h, x:x+w, 1] = alt_img[y:y+h, x:x+w, 1] * (1-mask) + mask * gen_img[:, :, 1]  #*255.0
            #alt_img[y:y+h, x:x+w, 2] = alt_img[y:y+h, x:x+w, 2] * (1-mask) + mask * gen_img[:, :, 2]  #*255.0
        except:
            import sys
            print("ERROR: replace - unexpected error:", sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2])
            #print("ERROR: {}, {}, {}".format(gen_img[:, :, 0].shape, alt_img[y:y+h, x:x+w, 0].shape, mask.shape))
            raise

    dest_path = './demo.png'
    #imsave(os.path.join(dest_path, item), img)
    imsave(dest_path, alt_img)