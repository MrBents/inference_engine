import os
# os.environ['KERAS_BACKEND'] = 'theano'
import numpy as np
from skimage import color, exposure, feature, filters, io
from flask import Flask, request
from keras.models import load_model
import cv2
from keras import backend as K
import keras
K.set_image_dim_ordering('th')


def reshape_input_image_for_cnn(X):
    print(X.shape)
    img_rows = X.shape[0]
    img_cols = X.shape[1]
    if K.image_data_format() == 'channels_first':
        X = X.reshape(1, 1, img_rows, img_cols)
    else:
        X = X.reshape(1, img_rows, img_cols, 1)
    return X


def extract_image_features(data):
    feature_data = []
    for i in range(data.shape[0]):
        img = data[i]
        # img = color.rgb2gray(img)
        img = exposure.adjust_gamma(img, 2)
        kernel1 = np.ones((2, 2), np.uint8)
        gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel1)
        # gradient = gradient.flatten()
        feature_data.append(gradient)
    feature_data = np.array(feature_data)
    return(feature_data)


model = load_model(
    '/home/alfred/Desktop/zippers/models/HEADS_0.9929_CNN.h5')

image_top1 = cv2.imread(
    '/home/alfred/Desktop/zippers/image_data/all_top_images/IMG_8_bad_12cm_0.JPEG', 0)
image_top2 = cv2.imread(
    '/home/alfred/Desktop/zippers/image_data/all_top_images/IMG_8_good_12cm_0.JPEG', 0)
image_head1 = cv2.imread(
    '/home/alfred/Desktop/zippers/image_data/all_heads/IMG_2_badhead.JPEG', 0)
image_head2 = cv2.imread(
    '/home/alfred/Desktop/zippers/image_data/all_heads/IMG_2_goodhead.JPEG', 0)

image_head1 = extract_image_features(image_head1)
image_top1 = extract_image_features(image_top1)

image_top1 = reshape_input_image_for_cnn(image_top1)
image_head1 = reshape_input_image_for_cnn(image_head1)

print(model.predict_classes(image_head1))
