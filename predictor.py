import scipy
import os
# os.environ['KERAS_BACKEND'] = 'theano'
import numpy as np
from skimage import color, exposure, feature, filters, io
from flask import Flask, request
from keras.models import load_model
import cv2
from keras import backend as K
import tensorflow as tf
import keras
import binascii
import io
import json
global count


K.set_image_data_format('channels_last')

class Predictor:
    def __init__(self):
        record = ModelRecords().records
        thresh = ModelRecords().thresholds
        self.mapper = {}
        for key in record.keys():
            self.mapper[key] = Model(record[key], thresh[key])
        self.graph = tf.get_default_graph()

    def predict(self, img_head_path, img_top_path, z_type):
        # if z_type == 'max' or z_type == 'speed':
        # head_model = self.mapper['max45heads']
        # top_model = self.mapper['max45tops']
        # else:
        with self.graph.as_default():
            head_model = self.mapper['42heads']
            top_model = self.mapper['42tops']

            # head = head_model.predict(img_head_path)

            head_pred = []
            mini = int(round(img_head_path.shape[1]*0.03))
            maxi = int(round(img_head_path.shape[1]*0.25))
            for _ in range(0, 4):
                random_pixel_value = np.random.randint(mini, maxi)
                pos_neg = np.random.randint(0, 2)
                if (pos_neg == 0):
                    random_pixel_value = -random_pixel_value
                new_picture = scipy.ndimage.interpolation.shift(
                    img_head_path, (0, random_pixel_value), mode='nearest', order=5)
                new_picture_t = self.reshape_image_for_cnn(new_picture, 'head')
                prediction = head_model.predict(new_picture_t)
                # print(prediction)
                # value = self.get_value(prediction)
                head_pred.append(prediction)
            head_pred.append(head_model.predict(
                self.reshape_image_for_cnn(img_head_path, 'head')))
            head = int(np.round(np.array(head_pred).mean()))

            # if (head == 0):
            #     head = 1
            # elif (head == 1):
            #     head = 0
            top = top_model.predict(
                self.reshape_image_for_cnn(img_top_path, 'top'))
            # if (top == 0):
            #     top = 1
            # elif (top == 1):
            #     top = 0
            final_result = head * top
            print((head, top, final_result))
            return head, top, final_result

    def reshape_image_for_cnn(self, X, img_t):
        if img_t == 'head':
            X = cv2.resize(X, (320, 240))
        elif img_t == 'top':
            #X = X[0:, 174:-240]
            pass
        X = X.astype('float32')
        X /= 255
        num_of_images = 1
        img_rows = X.shape[0]
        img_cols = X.shape[1]
        X = X.reshape(num_of_images, img_rows, img_cols, 1)
        return X

    def get_value(self, p):
        val = p[0][0]
        if (val > 0.50):
            var = 0
        else:
            var = 1
        return var


class Model:
    def __init__(self, path, threshold):
        self.model = load_model(path)
        self.th = threshold

    def predict(self, image):
        pred = self.model.predict(image)
        if (pred[0][1] > self.th):
            res = 1
        else:
            res = 0

        # pred = np.argmax(pred, 1)
        # return (pred[0])
        return res


class ModelRecords:
    def __init__(self):
        # self.records = {'max45heads': '/home/enigma/Desktop/models/MobV2_FINAL2_HEADS_CNN_NEW33.h5',
        #                 'max45tops': '/home/enigma/Desktop/models/Mobilenet__FINAL_TOPS_CNN.h5'}
        # self.records = {'42heads': '/home/enigma/Desktop/models/MobV2_FINAL2_HEADS_42.h5',
        #                 '42tops': '/home/enigma/Desktop/models/MobV2_FINAL_TOPS_42_5.h5'}
        self.records = {'42heads': '/home/enigma/Desktop/models/heads_mobv1_42_new_3.5.h5',
                        '42tops': '/home/enigma/Desktop/models/MobV2_FINAL_TOPS_42_5.h5'}
        self.thresholds = {'42heads': 0.98, '42tops': 0.99}


predictor = Predictor()