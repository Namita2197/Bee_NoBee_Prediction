########################################################
# module: load_nets.py
# Namita Raghuvanshi
# A02310449
# descrption: starter code for loading your project 1 nets.
########################################################

import pickle
import numpy as np
import tensorflow as tf
import tflearn
from tflearn import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression

### ======================= ANNs ===========================

#Accuracy- 0.5608695652173913
def load_ann_audio_model_buzz1(model_path):
    input_layer = input_data(shape=[None, 4000, 1, 1])
    fc_layer_1 = fully_connected(input_layer, 2048,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 1024,
                                 activation='relu',
                                 name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 512,
                                 activation='relu',
                                 name='fc_layer_3')
    fc_layer_4 = fully_connected(fc_layer_3, 512,
                                 activation='relu',
                                 name='fc_layer_4')
    fc_layer_5 = fully_connected(fc_layer_4, 128,
                                 activation='relu',
                                 name='fc_layer_5')
    fc_layer_6 = fully_connected(fc_layer_5, 3,
                                 activation='softmax',
                                 name='fc_layer_6')
    model = tflearn.DNN(fc_layer_6)
    model.load(model_path)
    return model

#Accuracy - 0.5126
def load_ann_audio_model_buzz2(model_path):
    input_layer = input_data(shape=[None, 4000, 1, 1])
    fc_layer_1 = fully_connected(input_layer, 1600,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 1200,
                                 activation='relu',
                                 name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 200,
                                 activation='relu',
                                 name='fc_layer_3')
    fc_layer_6 = fully_connected(fc_layer_3, 3,
                                 activation='softmax',
                                 name='fc_layer_6')
    model = tflearn.DNN(fc_layer_6)
    model.load(model_path)
    return model

#Accuracy - 0.6921322690992018
def load_ann_audio_model_buzz3(model_path):
    input_layer = input_data(shape=[None, 4000, 1, 1])
    fc_layer_1 = fully_connected(input_layer, 800,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 400,
                                 activation='relu',
                                 name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 400,
                                 activation='relu',
                                 name='fc_layer_3')
    fc_layer_4 = fully_connected(fc_layer_3, 200,
                                 activation='relu',
                                 name='fc_layer_4')
    fc_layer_5 = fully_connected(fc_layer_4, 90,
                                 activation='relu',
                                 name='fc_layer_5')
    fc_layer_6 = fully_connected(fc_layer_5, 3,
                                 activation='softmax',
                                 name='fc_layer_6')
    model = tflearn.DNN(fc_layer_6)
    model.load(model_path)
    return model

#Accuracy - 0.9013605442176871
def load_ann_image_model_bee1_gray(model_path):
    input_layer = input_data(shape=[None, 64, 64, 1])
    fc_layer_1 = fully_connected(input_layer, 512,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 256,
                                 activation='relu',
                                 name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 128,
                                 activation='relu',
                                 name='fc_layer_3')
    fc_layer_4 = fully_connected(fc_layer_3, 64,
                                 activation='relu',
                                 name='fc_layer_4')
    fc_layer_5 = fully_connected(fc_layer_4, 2,
                                 activation='softmax',
                                 name='fc_layer_5')
    model = tflearn.DNN(fc_layer_5)
    model.load(model_path)
    return model

#Accuracy - 0.7480846406421015
def load_ann_image_model_bee2_1s_gray(model_path):
    input_layer = input_data(shape=[None, 64, 64, 1])
    fc_layer_1 = fully_connected(input_layer, 700,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 400,
                                 activation='relu',
                                 name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 200,
                                 activation='relu',
                                 name='fc_layer_3')
    fc_layer_4 = fully_connected(fc_layer_3, 2,
                                 activation='softmax',
                                 name='fc_layer_4')
    model = tflearn.DNN(fc_layer_4)
    model.load(model_path)
    return model

#Accuracy - 0.7892168972332015
def load_ann_image_model_bee4_gray(model_path):
    input_layer = input_data(shape=[None, 64, 64, 1])
    fc_layer_1 = fully_connected(input_layer, 700,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 350,
                                 activation='relu',
                                 name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 2,
                                 activation='softmax',
                                 name='fc_layer_3')
    model = tflearn.DNN(fc_layer_3)
    model.load(model_path)
    return model

### ======================= ConvNets ===========================

#Accuracy - 0.6373913043478261
def load_cnn_audio_model_buzz1(model_path):
    input_layer = input_data(shape=[None, 4000, 1, 1])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=20,
                           filter_size=8,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                           nb_filter=10,
                           filter_size=8,
                           activation='relu',
                           name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    fc_layer_1 = fully_connected(pool_layer_2, 120,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 3,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(model_path)
    return model

#Accuracy - 0.5633333333333334
def load_cnn_audio_model_buzz2(model_path):
    input_layer = input_data(shape=[None, 4000, 1, 1])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=12,
                           filter_size=6,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                           nb_filter=10,
                           filter_size=6,
                           activation='relu',
                           name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    conv_layer_3 = conv_2d(pool_layer_2,
                           nb_filter=8,
                           filter_size=6,
                           activation='relu',
                           name='conv_layer_3')
    pool_layer_3 = max_pool_2d(conv_layer_3, 2, name='pool_layer_3')

    fc_layer_1 = fully_connected(pool_layer_3, 128,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 64,
                                 activation='relu',
                                 name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 3,
                                 activation='softmax',
                                 name='fc_layer_3')
    model = tflearn.DNN(fc_layer_3)
    model.load(model_path)
    return model

# Accuracy - 0.9341505131128849
def load_cnn_audio_model_buzz3(model_path):
    input_layer = input_data(shape=[None, 4000, 1, 1])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=20,
                           filter_size=8,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                           nb_filter=10,
                           filter_size=8,
                           activation='relu',
                           name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    fc_layer_1 = fully_connected(pool_layer_2, 120,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 3,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(model_path)
    return model

#Accuracy - 0.9713718820861678
def load_cnn_image_model_bee1(model_path):
    input_layer = input_data(shape=[None, 64, 64, 3])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=4,
                           filter_size=4,
                           strides=2,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                           nb_filter=4,
                           filter_size=6,
                           strides=2,
                           activation='relu',
                           name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    fc_layer_1 = fully_connected(pool_layer_2, 128,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(model_path)
    return model

#Accuracy - 0.9062385990514411
def load_cnn_image_model_bee2_1s(model_path):
    input_layer = input_data(shape=[None, 64, 64, 3])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=12,
                           filter_size=4,
                           strides=2,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                           nb_filter=6,
                           filter_size=4,
                           strides=2,
                           activation='relu',
                           name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    fc_layer_1 = fully_connected(pool_layer_2, 100,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(model_path)
    return model

#Accuracy - 0.7914402173913043
def load_cnn_image_model_bee4(model_path):
    input_layer = input_data(shape=[None, 64, 64, 3])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=9,
                           filter_size=4,
                           strides=2,
                           activation='tanh',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                           nb_filter=6,
                           filter_size=4,
                           strides=2,
                           activation='tanh',
                           name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    conv_layer_3 = conv_2d(pool_layer_2,
                           nb_filter=3,
                           filter_size=4,
                           strides=2,
                           activation='tanh',
                           name='conv_layer_3')
    pool_layer_3 = max_pool_2d(conv_layer_3, 2, name='pool_layer_3')
    fc_layer_1 = fully_connected(pool_layer_3, 64,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(model_path)
    return model


