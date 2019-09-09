import os
import numpy as np
import cv2

# include demo dataset mnist
#from keras.datasets import mnist

from scipy.misc import imresize, imsave #, imread, imsave
from scipy.ndimage import imread
from skimage.transform import resize, rotate

from utils import trim, read_files, psnr, log10, preprocess_size, \
     loadAndResizeImages2, preprocess_enhance_edges
from utils import load_parameters

try:
    import imgaug as ia
    from imgaug import augmenters as iaa
except:
    ia = None
    iaa = None

def cdist(a, b, keepdims=False):
    d = np.sqrt(np.sum(np.square(a - b), axis=2))
    if keepdims:
        d = np.repeat(d[:,:,np.newaxis], 3, axis=2)
    return d
    
max_cdist = cdist(np.zeros((1, 1, 3)), np.ones((1, 1, 3)))

#def cdist_average(avg_color, b, keepdims=False):
#    avg_img = np.ones(b.shape) * avg_color
#    return cdist(avg_img, b, keepdims=False)

def random_rotation(image_array: np.ndarray, angle=None):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    if angle is None:
        angle = np.random.uniform(-180, 180)
    return rotate(image_array, angle)


def brownian_data_generator(dir_with_src_images, base_image_filename, object_image_list, img_shape=(28, 28, 1), batch_size=32, resized_objects=None):

    h, w, ch = img_shape

    # define inputs
    batch_inputs = np.zeros((batch_size, h, w, ch), dtype=np.float32)
    # define outputs
    batch_outputs = np.zeros((batch_size, h, w, ch), dtype=np.float32)
    # define attention masks (by default ones as everything has the same importance)
    batch_masks = np.ones((batch_size, h, w, ch), dtype=np.float32)

    
    def preprocess_size_helper(new_dim=(h, w)):
        return lambda image: preprocess_size(image, new_dim)
            
    preprocessors = [preprocess_size_helper(new_dim=(h, w)), preprocess_enhance_edges]

    # load images
    base_image = loadAndResizeImages2(dir_with_src_images, [base_image_filename])[0]
    objects = loadAndResizeImages2(dir_with_src_images, object_image_list, load_alpha=True)

    # load params, since some of them are needed to generate data:
    parameters_filepath = "config.ini"
    parameters = load_parameters(parameters_filepath)

    size_factor = float(parameters["synthetic"]["size_factor"])
    save_batch = eval(parameters["synthetic"]["save_batch"])
    calc_masks = eval(parameters["synthetic"]["calc_masks"])
    dilate_masks = int(parameters["synthetic"]["dilate_masks"])
    blur_masks = eval(parameters["synthetic"]["blur_masks"])
    blur_kernel = eval(parameters["synthetic"]["blur_kernel"])
    
    obj_attention = float(parameters["synthetic"]["obj_attention"])
    back_attention = float(parameters["synthetic"]["back_attention"])

    subtract_median = eval(parameters["synthetic"]["subtract_median"])

    add_noise = eval(parameters["synthetic"]["add_noise"])
    noise_amnt = float(parameters["synthetic"]["noise_amnt"])

    loss = (parameters["hyperparam"]["loss"])

    # median threshold
    threshold = .5

    # resize to desired size
    orig_h, orig_w, _ = base_image.shape
    ratio_h = orig_h / h
    ratio_w = orig_w / w
    
    base_image = preprocess_size(base_image, (h, w))
    if resized_objects is None:
        resized_objects = []
    objects_pos = []
    rotated_objs = [None] * len(objects)
    for o in objects:
        ho, wo, cho = o.shape
        if ho == wo:
            hn = int((ho / ratio_w)*size_factor)
            wn = int((wo / ratio_w)*size_factor)
        else:
            hn = int((ho / ratio_h)*size_factor)
            wn = int((wo / ratio_w)*size_factor)
        resized_o = preprocess_size(o, (hn, wn))
        resized_objects.append(resized_o)

        #
        #print(w, " - ", wo)
        x = np.random.randint(low=0, high=w-wn) # +wo
        y = np.random.randint(low=(60 / ratio_h), high=h-hn-(30 / ratio_h)) 
        objects_pos.append((x, y))
    
    L = wn // 2
    #print("L: ", L)
    a = 0
    replace_robots = True
    P_greed = 1. / 33. #1 / 20. #1 / 33.
   
    iteration = -1
    
    # serve randomly generated images
    while True:
        iteration += 1
        # go through the entire dataset, using batch_sized chunks each time
        
        batch_actions = np.zeros((batch_size, 1), dtype=np.float32)
        batch_rewards = np.zeros((batch_size, 1), dtype=np.float32)

        for i in range(0, batch_size):
            
            np.copyto(batch_inputs[i], base_image)

            # TODO: randomly place the objects:
            if replace_robots:
                for ix, o in enumerate(resized_objects):
                    a = np.random.uniform(-180, 180)
                    o_rot = random_rotation(o, angle=a-90) # +90 since robot is sideways
                    #if ix == 1:
                    #    batch_actions[i] = a / 180
                    ho, wo, cho = o_rot.shape
                    
                    x = np.random.randint(low=0, high=w-wo) # +wo
                    #print((100 / ratio_h))
                    # 30 is the magic number to limit the random placement of objects inside image
                    y = np.random.randint(low=(60 / ratio_h), high=h-ho-(30 / ratio_h)) 

                    xg = x - (wo // 2)
                    yg = y - (ho // 2)

                    if xg + wo > w:
                        xg = w - wo
                    if yg + ho > h:
                        yg = h - ho
                    if xg < 0:
                        xg = 0
                    if yg < 0:
                        yg = 0

                    x = xg + (wo // 2)
                    y = yg + (ho // 2)

                    objects_pos[ix] = (x, y)
                    rotated_objs[ix] = o_rot 

                replace_robots = False
            else:
                # move the robot into random orientation with fixed direction 
                #imsave("tmp/{}.png".format("obj_generated_" + str(i)),  o_rot)
                robot_object_distance = h+w
                for ix, o in enumerate(resized_objects):
                    if ix == 1: # only do this for the robot and not the object
                        
                        x, y = objects_pos[ix]
                        x_t, y_t = objects_pos[ix-1]
                        # select angle towards the object in 10% of the time
                        if np.random.random() <= P_greed:
                            a = np.arctan2(x_t - x, y_t - y)
                            a = np.degrees(a) # -180 <> 180
                            #print("target angle: ", a)
                        else: # select angle randomly in 90% of the time
                            a = np.random.uniform(-180, 180)
                            #a = a * 360 # -180 <> 180
                        
                        action = a # action in degrees

                        
                        o_rot = random_rotation(o, a-90) # +90 since robot is sideways
                        ho, wo, cho = o_rot.shape
                        
                        # this is new position of the robot
                        a = np.radians(a) # need to convert to radians before using sin and cos
                        batch_actions[i] = action / 180 #np.sin(a), np.cos(a) # action / 180
                        
                        x = x + L * np.sin(a)
                        y = y + L * np.cos(a)

                        x = int(np.round(x))
                        y = int(np.round(y))
                        
                        xg = x - (wo // 2)
                        yg = y - (ho // 2)

                        if xg + wo > w:
                            xg = w - wo
                        if yg + ho > h:
                            yg = h - ho
                        if xg < 0:
                            xg = 0
                        if yg < 0:
                            yg = 0

                        x = xg + (wo // 2)
                        y = yg + (ho // 2)

                        objects_pos[ix] = (x, y)
                        rotated_objs[ix] = o_rot 

                        robot_object_distance = np.sqrt((x-x_t)**2 + (y-y_t)**2)
                        #print("robot dist / L: {} / {}".format(robot_object_distance, L))

                if robot_object_distance < L:
                    reward = 1
                    batch_rewards[i] = reward
                    replace_robots = True

            for ix, o in enumerate(resized_objects):
                x, y = objects_pos[ix]
                o_rot = rotated_objs[ix]
                ho, wo, cho = o_rot.shape
                mask = o_rot[:, :, 3] # / 255.0
                #print(mask.max(), mask.min())
                #print(x, y, ho, wo, cho)
                xg = x - (wo // 2)
                yg = y - (ho // 2)


                batch_inputs[i][yg:yg+ho, xg:xg+wo, 0] = batch_inputs[i][yg:yg+ho, xg:xg+wo, 0] * (1-mask) + mask * o_rot[:, :, 0] #*255.0
                batch_inputs[i][yg:yg+ho, xg:xg+wo, 1] = batch_inputs[i][yg:yg+ho, xg:xg+wo, 1] * (1-mask) + mask * o_rot[:, :, 1] #*255.0
                batch_inputs[i][yg:yg+ho, xg:xg+wo, 2] = batch_inputs[i][yg:yg+ho, xg:xg+wo, 2] * (1-mask) + mask * o_rot[:, :, 2] #*255.0
                
            #imsave("tmp/{}.png".format("in_generated_" + str(i)),  batch_inputs[i])
            np.copyto(batch_outputs[(i-1)%batch_size], batch_inputs[i])
            #imsave("tmp/{}.png".format("out_generated_" + str(i)),  batch_outputs[i])
            #print(batch_outputs[i].max(), batch_outputs[i].min())
        batch_median = np.median(batch_outputs, axis=0, keepdims=True) 
        
        #print("Batch median shape: ", batch_median.shape)
        #print("Batch median shape: ", batch_outputs.shape)
        if calc_masks:
            median_min = batch_median[0].min()
            median_max = batch_median[0].max()
            for i in range(0, batch_size):
                
                tmp = cdist(batch_median[0], batch_inputs[i], keepdims=True) # color distance between images
                mask = (tmp > threshold*max_cdist).astype(float)
                batch_masks[i] = mask * obj_attention
                #back_mask = ( tmp <= 0 ).astype(float) + back_attention

                #batch_masks[i][batch_masks[i] > 0.5] += 0.1
                # uncomment to blur the images (soft attention)
                if dilate_masks > 0:
                    #print("dilating masks...")
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
                    batch_masks[i] = cv2.dilate(batch_masks[i], kernel, iterations=dilate_masks) 
                if back_attention > 0.0:
                    #print("Setting background weights...")
                #    back_mask = ( tmp <= 0 ).astype(float) + back_attention
                    batch_masks[i] += ((1 - (mask).astype(int)).astype(float) * back_attention)
                    
                if blur_masks:
                    #print("Blurring masks....")
                    batch_masks[i] = cv2.blur(batch_masks[i], blur_kernel) # add blur if needed
                
                if save_batch: # save generated images to tmp folder
                    me_min = batch_actions[i][0] # batch_masks[i].min()
                    me_max = batch_rewards[i][0] #batch_masks[i].max()
                    label = str(i)#str(np.random.randint(0, 1000))
                    imsave("tmp/{}.png".format("m_{}_{}_{:01.2f}_{}".format(iteration, label, me_min, me_max)), batch_masks[i])
                    imsave("tmp/{}.png".format("A_{}_{}_{:01.2f}_{}".format(iteration, label, me_min, me_max)), batch_inputs[i])
                    imsave("tmp/{}.png".format("B_{}_{}_{:01.2f}_{}".format(iteration, label, me_min, me_max)), batch_outputs[i])
                    
            if save_batch: # save only first batch
                save_batch = False
            
        #batch_percentile = np.percentile(batch_outputs, 99.9, axis=0, keepdims=True)
        #label = str(np.random.randint(0, 1000))
        #imsave("tmp/{}.png".format("percentile_99.9_" + str(label)), batch_percentile[0])
        if subtract_median:
            #batch_mean = batch_outputs.mean(axis=0, keepdims=True)
            # careful - batch_size must be greater than 1!!!
            #batch_median = np.median(batch_outputs, axis=0, keepdims=True) 
            
            #imsave("tmp/{}.png".format("median_" + str(i)), batch_median[0])
            batch_outputs = batch_median - batch_outputs
            #imsave("tmp/{}.png".format("out1_" + str(i)), batch_outputs[0])
            
        if add_noise:
            batch_inputs += noise_amnt * np.random.normal(loc=0.0, scale=1.0, size=batch_inputs.shape) 
        
        #label = str(np.random.randint(0, 1000))
        #imsave("tmp/{}.png".format(label + "_in_generated_" + str(i)),  batch_inputs[0])
        #imsave("tmp/{}.png".format(label + "_out_generated_" + str(i)),  batch_median[0] - batch_outputs[0])
        #print(batch_median.shape)
        #if 'wmse' in loss and 'out-median' in mode:
        #    yield [batch_inputs, np.repeat(np.array([batch_median]), batch_size, axis=0).reshape((batch_size, h, w, 3))], batch_outputs
        #print(batch_actions.shape)
        if 'wmse' in loss:
            yield [batch_inputs, batch_masks, batch_actions], batch_outputs # , batch_actions, batch_masks
        else:
            yield [batch_inputs, batch_actions], batch_outputs

def random_data_generator(dir_with_src_images, base_image_filename, object_image_list, img_shape=(28, 28, 1), batch_size=32, resized_objects=None):

    h, w, ch = img_shape

    # define inputs
    batch_inputs = np.zeros((batch_size, h, w, ch), dtype=np.float32)
    # define outputs
    batch_outputs = np.zeros((batch_size, h, w, ch), dtype=np.float32)
    # define attention masks (by default ones as everything has the same importance)
    batch_masks = np.ones((batch_size, h, w, ch), dtype=np.float32)

    def preprocess_size_helper(new_dim=(h, w)):
        return lambda image: preprocess_size(image, new_dim)
            
    preprocessors = [preprocess_size_helper(new_dim=(h, w)), preprocess_enhance_edges]

    # load images
    base_image = loadAndResizeImages2(dir_with_src_images, [base_image_filename])[0]
    objects = loadAndResizeImages2(dir_with_src_images, object_image_list, load_alpha=True)

    # load params, since some of them are needed to generate data:
    parameters_filepath = "config.ini"
    parameters = load_parameters(parameters_filepath)

    size_factor = float(parameters["synthetic"]["size_factor"])
    save_batch = eval(parameters["synthetic"]["save_batch"])
    calc_masks = eval(parameters["synthetic"]["calc_masks"])
    dilate_masks = int(parameters["synthetic"]["dilate_masks"])
    blur_masks = eval(parameters["synthetic"]["blur_masks"])
    blur_kernel = eval(parameters["synthetic"]["blur_kernel"])
    
    obj_attention = float(parameters["synthetic"]["obj_attention"])
    back_attention = float(parameters["synthetic"]["back_attention"])

    subtract_median = eval(parameters["synthetic"]["subtract_median"])

    add_noise = eval(parameters["synthetic"]["add_noise"])
    noise_amnt = float(parameters["synthetic"]["noise_amnt"])

    loss = (parameters["hyperparam"]["loss"])

    # median threshold
    threshold = .5

    # resize to desired size
    orig_h, orig_w, _ = base_image.shape
    ratio_h = orig_h / h
    ratio_w = orig_w / w
    
    base_image = preprocess_size(base_image, (h, w))
    if resized_objects is None:
        resized_objects = []
    for o in objects:
        ho, wo, cho = o.shape
        if ho == wo:
            hn = int((ho / ratio_w)*size_factor)
            wn = int((wo / ratio_w)*size_factor)
        else:
            hn = int((ho / ratio_h)*size_factor)
            wn = int((wo / ratio_w)*size_factor)
        resized_o = preprocess_size(o, (hn, wn))
        resized_objects.append(resized_o)
    
   
    # serve randomly generated images
    while True:
        
        # go through the entire dataset, using batch_sized chunks each time
        for i in range(0, batch_size):
            
            np.copyto(batch_inputs[i], base_image)

            # TODO: randomly place the objects:
            for o in resized_objects:
                o_rot = random_rotation(o)

                ho, wo, cho = o_rot.shape
                
                x = np.random.randint(low=0, high=w-wo) # +wo
                #print((100 / ratio_h))
                # 30 is the magic number to limit the random placement of objects inside image
                y = np.random.randint(low=(60 / ratio_h), high=h-ho-(30 / ratio_h)) 

                #imsave("tmp/{}.png".format("obj_generated_" + str(i)),  o_rot)

                mask = o_rot[:, :, 3] # / 255.0
                #print(mask.max(), mask.min())
                batch_inputs[i][y:y+ho, x:x+wo, 0] = batch_inputs[i][y:y+ho, x:x+wo, 0] * (1-mask) + mask * o_rot[:, :, 0] #*255.0
                batch_inputs[i][y:y+ho, x:x+wo, 1] = batch_inputs[i][y:y+ho, x:x+wo, 1] * (1-mask) + mask * o_rot[:, :, 1] #*255.0
                batch_inputs[i][y:y+ho, x:x+wo, 2] = batch_inputs[i][y:y+ho, x:x+wo, 2] * (1-mask) + mask * o_rot[:, :, 2] #*255.0
                
            #imsave("tmp/{}.png".format("in_generated_" + str(i)),  batch_inputs[i])
            np.copyto(batch_outputs[i], batch_inputs[i])
            #imsave("tmp/{}.png".format("out_generated_" + str(i)),  batch_outputs[i])
            #print(batch_outputs[i].max(), batch_outputs[i].min())
        batch_median = np.median(batch_outputs, axis=0, keepdims=True) 
        
        #print("Batch median shape: ", batch_median.shape)
        #print("Batch median shape: ", batch_outputs.shape)
        if calc_masks:
            median_min = batch_median[0].min()
            median_max = batch_median[0].max()
            for i in range(0, batch_size):
                
                tmp = cdist(batch_median[0], batch_inputs[i], keepdims=True) # color distance between images
                mask = (tmp > threshold*max_cdist).astype(float)
                batch_masks[i] = mask * obj_attention
                #back_mask = ( tmp <= 0 ).astype(float) + back_attention

                #batch_masks[i][batch_masks[i] > 0.5] += 0.1
                # uncomment to blur the images (soft attention)
                if dilate_masks > 0:
                    #print("dilating masks...")
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
                    batch_masks[i] = cv2.dilate(batch_masks[i], kernel, iterations=dilate_masks) 
                if back_attention > 0.0:
                    #print("Setting background weights...")
                #    back_mask = ( tmp <= 0 ).astype(float) + back_attention
                    batch_masks[i] += ((1 - (mask).astype(int)).astype(float) * back_attention)
                    
                if blur_masks:
                    #print("Blurring masks....")
                    batch_masks[i] = cv2.blur(batch_masks[i], blur_kernel) # add blur if needed
                
                if save_batch: # save generated images to tmp folder
                    me_min = batch_masks[i].min()
                    me_max = batch_masks[i].max()
                    label = str(np.random.randint(0, 1000))
                    imsave("tmp/{}.png".format("mask_{}_{}_{}".format(label, me_min, me_max)), batch_masks[i])
                    imsave("tmp/{}.png".format("input_{}_{}_{}".format(label, me_min, me_max)), batch_inputs[i])
                    
            if save_batch: # save only first batch
                save_batch = False
            
        #batch_percentile = np.percentile(batch_outputs, 99.9, axis=0, keepdims=True)
        #label = str(np.random.randint(0, 1000))
        #imsave("tmp/{}.png".format("percentile_99.9_" + str(label)), batch_percentile[0])
        if subtract_median:
            #batch_mean = batch_outputs.mean(axis=0, keepdims=True)
            # careful - batch_size must be greater than 1!!!
            #batch_median = np.median(batch_outputs, axis=0, keepdims=True) 
            
            #imsave("tmp/{}.png".format("median_" + str(i)), batch_median[0])
            batch_outputs = batch_median - batch_outputs
            #imsave("tmp/{}.png".format("out1_" + str(i)), batch_outputs[0])
            
        if add_noise:
            batch_inputs += noise_amnt * np.random.normal(loc=0.0, scale=1.0, size=batch_inputs.shape) 
        
        #label = str(np.random.randint(0, 1000))
        #imsave("tmp/{}.png".format(label + "_in_generated_" + str(i)),  batch_inputs[0])
        #imsave("tmp/{}.png".format(label + "_out_generated_" + str(i)),  batch_median[0] - batch_outputs[0])
        #print(batch_median.shape)
        #if 'wmse' in loss and 'out-median' in mode:
        #    yield [batch_inputs, np.repeat(np.array([batch_median]), batch_size, axis=0).reshape((batch_size, h, w, 3))], batch_outputs
        if 'wmse' in loss:
            yield [batch_inputs, batch_masks], batch_outputs
        else:
            yield batch_inputs, batch_outputs

def data_generator(dir_with_src_images, img_list, img_shape=(28, 28, 1), batch_size=32, mode=''):

    h, w, ch = img_shape

    # define inputs
    batch_inputs = np.zeros((batch_size, h, w, ch), dtype=np.float32)
    # define outputs
    batch_outputs = np.zeros((batch_size, h, w, ch), dtype=np.float32)

    def preprocess_size_helper(new_dim=(h, w)):
        return lambda image: preprocess_size(image, new_dim)
            
    preprocessors = [preprocess_size_helper(new_dim=(h, w))]

    # check if batch_size divides the size of entire dataset properly
    if len(img_list) % batch_size != 0:
        remain = batch_size - (len(img_list) % batch_size)
        #to_fill = remain - batch_size
        img_list = img_list + img_list[0:remain]
        print("REMAINDER:  ", len(img_list) % batch_size, " ", len(img_list[0:remain]))
    else:
        print("REMAINDER: is zero!")
    
    while True:
        
        # go through the entire dataset, using batch_sized chunks each time
        for i in range(0, len(img_list), batch_size):
            batch_images = img_list[i:i+batch_size]
            # TODO: resize images if needed
            batch_inputs = loadAndResizeImages2(dir_with_src_images, batch_images, preprocessors)
            # subtract mean to prevent AE to learn about background
            
            np.copyto(batch_outputs, batch_inputs)
            
            batch_median = np.zeros(batch_outputs.shape)
            if batch_size > 1:
                batch_median = np.median(batch_outputs, axis=0, keepdims=True) 

            if 'sub-mean' in mode:
                #batch_mean = batch_outputs.mean(axis=0, keepdims=True)
                # careful - batch_size must be greater than 1!!!
                
                #print("mean MIN MAX: ", batch_median[:].min(), batch_median[:].max())
                #print("before MIN MAX: ", batch_outputs[:].min(), batch_outputs[:].max())
                imsave("tmp/{}.png".format("median_" + str(i)), batch_median[0])
                batch_outputs = batch_median - batch_outputs
                imsave("tmp/{}.png".format("out1_" + str(i)), batch_outputs[0])
                #print("after MIN MAX: ", batch_outputs[:].min(), batch_outputs[:].max())
                #input()
                #batch_outputs = np.abs(batch_outputs)
                
            if 'noisy' in mode:
                batch_inputs += 0.15 * np.random.normal(loc=0.0, scale=1.0, size=batch_inputs.shape) 

            #print("MAX MIN: ", np.max(batch_inputs[0][:]), " ", np.min(batch_inputs[0][:]))
            #input("Continue?")
           
            # TODO: use imgaug to augment the images
            if 'out-median' in mode:
                yield [batch_inputs, np.repeat(np.array([batch_median]), batch_size, axis=0).reshape((batch_size, h, w, 3))], batch_outputs
            else:
                yield batch_inputs, batch_outputs

            

def data_generator_mnist(x_train, x_test, img_shape=(28, 28, 1), isTrain = True, batch_size = 100):
    
    '''
    nb_classes = 10
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    '''
    if(isTrain):
        dataset = x_train
    else :
        dataset = x_test

    dataset_size = dataset.shape[0]

    h, w, ch = img_shape
    input_batch = np.zeros((batch_size, h, w, ch))
    output_batch = np.zeros((batch_size, h, w, ch))

    while(True):
        i = 0

        for j in range(batch_size):
            input_batch[j] = resize(dataset[i+j], (h, w), anti_aliasing=True)
            output_batch[j] = resize(dataset[i+j], (h, w), anti_aliasing=True)

        yield input_batch, output_batch

        i += batch_size
        if (i+batch_size>dataset_size):
            i = 0



if __name__ == '__main__':

    # test data generators if they work properly
    img_dim = 32
    img_shape = (480, 640, 3)
    
    img_shape = (240, 320, 3)
    img_shape = (128, 128, 3)
    img_shape = (64, 64, 3)
    dir_with_src_images = 'data\\generated\\'
    base_image = 'median_image.png'
    object_images = ['circle.png', 'robo.png'] # circle in the first place, as robo can be drawn over it
    object_images = ['circle-red.png', 'robo-green-a.png']
    fitting_generator = brownian_data_generator(dir_with_src_images, base_image, object_images, img_shape=img_shape)


    #fitting_generator.__next__()

    c = 0
    for x_in, x_out in fitting_generator:
        print(type(x_in), len(x_in))
        print(x_out.shape)
        c += 1
        if c > 5:
            break