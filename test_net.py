from __future__ import print_function
from __future__ import division
from sklearn.metrics import confusion_matrix
#import click
import json
import os
import numpy as np
import SimpleITK as sitk
import numpy as np
from loadData_all import load_data
from model import *
from Statistics import *
import tensorflow as tf
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

K.set_image_data_format('channels_last')
from skimage import data
from skimage.morphology import square
from skimage.filters import rank

from evaluation import getDSC, getHausdorff, getVS
from prettytable import PrettyTable
from cca_post import CCA_postprocessing
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
def get_eval_metrics(true_mask, pred_mask):
    true_mask_sitk = sitk.GetImageFromArray(true_mask)
    pred_mask_sitk = sitk.GetImageFromArray(pred_mask)
    dsc = getDSC(true_mask_sitk, pred_mask_sitk)
    h95 = getHausdorff(true_mask_sitk, pred_mask_sitk)
    #vs = getVS(true_mask_sitk, pred_mask_sitk)

    result = {}
    result['dsc'] = dsc
    result['h95'] = h95
    #result['vs'] = vs

    return dsc, h95 #(dsc, h95, vs)



model = D_Unet(5)
#model = Unet(5)

model.load_weights('saved_models/dual_input.h5')
train_list = ['7', '8_2', '9_4', '14', '20_2', '20', '23', '30']
val_list = ['8', '10_2', '10', '22_1', '31']

X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')

for ss in range(np.shape(X_val)[0]):
    pred_test=model.predict(X_val[ss:ss+1],verbose=1)
    pred =pred_test.argmax(axis=-1)
    g_true =y_val[ss:ss+1].argmax(axis=-1)
    print(np.shape(g_true))
    print(np.shape(pred))
    #pred_test_t=pred_test.argmax(axis=-1)
    pred_cca = []
    for ii in range(np.shape(pred)[0]):
        pred_cca.append(CCA_postprocessing(np.uint8(pred[ii, ...])))
    pred_cca = np.asarray(pred_cca)

    dsc, h95 = get_eval_metrics(g_true, pred_cca)
    #dsc = get_eval_metrics(g_true, pred_cca)

    #h95 = get_eval_metrics(true_mask, pred_mask)
    # dsc = get_eval_metrics(true_mask, pred_mask)
    print('dsc:'+str(dsc))
    print('h95:'+str(h95))

def confusion_matrix(y_train, y_pred):
    true_class_0=y_train[0,:,:,0].astype(int)
    true_class_1=y_train[0,:,:,1].astype(int)
    true_class_2=y_train[0,:,:,2].astype(int)
    true_class_3=y_train[0,:,:,3].astype(int)
    true_class_4 =y_train[0,:,:,4].astype(int)

    pre_class_0=y_pred[0,:,:,0].astype(int)
    pre_class_1 = y_pred[0, :, :, 1].astype(int)
    pre_class_2 = y_pred[0, :, :, 2].astype(int)
    pre_class_3 = y_pred[0, :, :, 3].astype(int)
    pre_class_4 = y_pred[0, :, :, 4].astype(int)

    x=PrettyTable(["confusion","Background","1.Layer","2.Layer","3.Layer","4.Layer"])

    x.add_row(
        ["Background", str(((true_class_0 * pre_class_0) == 1).sum()), str(((true_class_0 * pre_class_1) == 1).sum()),
         str(((true_class_0 * pre_class_2) == 1).sum()), str(((true_class_0 * pre_class_3) == 1).sum()),
         str(((true_class_0 * pre_class_4) == 1).sum())])



    x.add_row(
        ["1.Layer", str(((true_class_1 * pre_class_0) == 1).sum()), str(((true_class_1 * pre_class_1) == 1).sum()),
         str(((true_class_1 * pre_class_2) == 1).sum()), str(((true_class_1 * pre_class_3) == 1).sum()),
         str(((true_class_1 * pre_class_4) == 1).sum())])

    x.add_row(
        ["2.Layer", str(((true_class_2 * pre_class_0) == 1).sum()), str(((true_class_2 * pre_class_1) == 1).sum()),
         str(((true_class_2 * pre_class_2) == 1).sum()), str(((true_class_2 * pre_class_3) == 1).sum()),
         str(((true_class_2 * pre_class_4) == 1).sum())])

    x.add_row(
        ["3.Layer", str(((true_class_3 * pre_class_0) == 1).sum()), str(((true_class_3 * pre_class_1) == 1).sum()),
         str(((true_class_3 * pre_class_2) == 1).sum()), str(((true_class_3 * pre_class_3) == 1).sum()),
         str(((true_class_3 * pre_class_4) == 1).sum())])

    x.add_row(
        ["4.Layer", str(((true_class_4 * pre_class_0) == 1).sum()), str(((true_class_4 * pre_class_1) == 1).sum()),
         str(((true_class_4 * pre_class_2) == 1).sum()), str(((true_class_4 * pre_class_3) == 1).sum()),
         str(((true_class_4 * pre_class_4) == 1).sum())])


    print(x)


