import os 
import numpy as np
#from keras.losses import binary_crossentropy, mse
#from keras import backend as K

from utils import preprocess_size, load_parameters

def f(model_label):
    if 'vae' in model_label:
        return 1
    else:
        return 0

if __name__ == '__main__':
    parameters_filepath = "config.ini"
    parameters = load_parameters(parameters_filepath)
    img_shape = eval(parameters["dataset"]["img_shape"])
    h, w, ch = img_shape
    h = 12
    w = 16

    dir_with_tflogs = 'tf-log/'
    dir_with_results = 'snapshots/'

    result_files = os.listdir(dir_with_tflogs)
    #print(result_files)
    result_files.remove('.gitkeep')

    summary = np.zeros((len(result_files), 19))
    summary_format = '%d,%d,%1.2f,%1.2f,%1.1f,%3.2f,%2.2f,%1.6f,%1.2f,%1.6f,%1.2f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f'

    print(summary.shape)

    for i, ex_label in enumerate(result_files):
        metrics_path = os.path.join(dir_with_results, ex_label+'.csv')
        metrics = np.loadtxt(metrics_path, skiprows=1, delimiter=',')
        params = load_parameters(os.path.join(dir_with_results, ex_label+'.ini'))
        print(ex_label)
        #print(metrics.shape)
        #print(params)
        #pdf_size = os.path.getsize(os.path.join(dir_with_results, ex_label+'.pdf'))
        #pdf_size = (pdf_size // 1024) # to get kbs
        #summary[i][-1] = pdf_size
        last_row = metrics[-1][1:] # 4 values, loss, psnr, val-loss, val-psnr
        #print(last_row)
        # parameters
        size_factor = float(params["synthetic"]["size_factor"])
        obj_attention = float(params["synthetic"]["obj_attention"])
        back_attention = float(params["synthetic"]["back_attention"])
        latent_size = int(params["hyperparam"]["latent_size"])
        model_label = f(params["hyperparam"]["model_label"])
        robot_ratio = obj_attention / back_attention
        loss_ratio =  1.0 / ((68.0**2 / (640.0**2)) * size_factor * robot_ratio)
        #print(loss_ratio)

        params = [model_label, latent_size, back_attention, obj_attention, size_factor, robot_ratio, loss_ratio]

        for j, p in enumerate(params):
            summary[i][j] = p
        summary[i][len(params):len(params)+last_row.shape[0]] = last_row

        eval_metrics = np.loadtxt('snapshots/{}.eval'.format(ex_label)) # avg-iou, std-iou, avg-minRB, std-minRB
        print(len(params)+last_row.shape[0]-4)
        summary[i][len(params)+last_row.shape[0]-4:] = eval_metrics
        

    # sort by loss before saving
    ind = np.argsort( summary[:,5] )[::-1] # bigger values first
    summary = summary[ind]
    with open('summary.csv', 'w') as fi:
        fi.write("model,latent,back-att,obj-att,obj-size,robot-ratio,loss-ratio,loss,psnr,val-loss,val-psnr,avg-mseR,std-mseR,avg-mseB,std-mseB,avg-iou,std-iou,avg-minRB,std-minRB\n")
        np.savetxt(fi, summary, fmt=summary_format)
