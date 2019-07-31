import os
#os.environ['KERAS_BACKEND'] = 'theano'
import numpy as np
from skimage import color, exposure, feature, filters, io
from flask import Flask, request
from keras.models import load_model
import cv2
from keras import backend as K
import keras


def reshape_input_image_for_cnn(X, img_t):
    if img_t == 'head':
        X = cv2.resize(X, (320, 240))
    else:
        X = X[0:, 174:-240]

    X = X.astype('float32')
    X /= 255
    print(X.shape)
    num_of_images = 1
    img_rows = X.shape[0]
    img_cols = X.shape[1]
    X = X.reshape(num_of_images, img_rows, img_cols, 1)
    return X


K.set_image_dim_ordering('th')
app = Flask(__name__)


head_model = load_model(
    '/home/alfred/Desktop/zippers/image_data/Mobilenet_HEAD_COMPLETE_VALIDATION_CNN.h5')
top_model = load_model(
    '/home/alfred/Desktop/zippers/image_data/Mobilenet_TOPS1.0_CNN.h5')


image_top1 = cv2.imread(
    '/home/alfred/Desktop/zippers/factory/tagged_images/all_tops/IMG_4_good.JPEG', 0)
image_head1 = cv2.imread(
    '/home/alfred/Desktop/zippers/factory/tagged_images/all_heads/IMG_3_goodhead.JPEG', 0)
image_top1 = reshape_input_image_for_cnn(image_top1, 'top')
image_head1 = reshape_input_image_for_cnn(image_head1, 'head')

pred = head_model.predict(image_head1)
print(np.argmax(pred, 1))
