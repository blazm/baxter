

from keras.losses import binary_crossentropy, mse
from keras import backend as K

from models import weighted_mean_squared_error
from utils import preprocess_size, load_parameters, loadAndResizeImages2

def preprocess_size_helper(new_dim=(128, 160)):
    return lambda image: preprocess_size(image, new_dim)
    

if __name__ == '__main__':
    parameters_filepath = "config.ini"
    parameters = load_parameters(parameters_filepath)
    img_shape = eval(parameters["dataset"]["img_shape"])
    h, w, ch = img_shape
    h = 12
    w = 16

    dir_with_src_images = 'test/'
    batch_images = ['frame000001.jpg', 'frame000011.jpg']
    preprocessors = [preprocess_size_helper(new_dim=(h, w))]

    images = loadAndResizeImages2(dir_with_src_images, batch_images, preprocessors)

    y_true = images[0]
    y_pred = images[1]

    y_true = K.variable(y_true)
    y_pred = K.variable(y_pred)
    error_wmse = K.eval(weighted_mean_squared_error(y_true, y_pred))
    error_mse = K.eval(mse(y_true, y_pred))

    print("MSE error: ", error_mse.max(), error_mse.min(), error_mse.mean())
    print("WMSE error: ", error_wmse.max(), error_wmse.min(), error_wmse.mean())

    print("MSE error: ", error_mse)
    print("WMSE error: ", error_wmse)
    #[ 0.69314718  0.91629082  0.35667494  0.22314353]