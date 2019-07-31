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

# K.set_image_dim_ordering('th')
K.set_image_data_format('channels_last')
app = Flask(__name__)


class Predictor:
    def __init__(self):
        record = ModelRecords().records
        thresh = ModelRecords().thresholds
        self.mapper = {}
        for key in record.keys():
            self.mapper[key] = Model(record[key], thresh[key])
        self.graph = tf.get_default_graph()
        self.curr_mod = ''
        

    def predict(self, img_head_path, img_top_path, z_type):
        # if z_type == 'max' or z_type == 'speed':
        # head_model = self.mapper['max45heads']
        # top_model = self.mapper['max45tops']
        # else:
        with self.graph.as_default():
            if '45' in z_type or 'max' in z_type and self.curr_mod != 'max45':
                head_model = self.mapper['max45heads']
                top_model = self.mapper['max45tops']
                self.curr_mod = 'max45'
            elif self.curr_mod != '42':
                self.curr_mod = '42'
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
            X = X[0:, 174:-240]
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
        if (pred[0][0] > self.th):
            res = 0
        else:
            res = 1

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
                        '42tops': '/home/enigma/Desktop/models/MobV2_FINAL_TOPS_42_5.h5',
                        'max45heads': '/home/enigma/Desktop/models/MobV2_FINAL2_HEADS_CNN_NEW33.h5',
                        'max45tops': '/home/enigma/Desktop/models/Mobilenet__FINAL_TOPS_CNN.h5'}
        self.thresholds = {'42heads': 0.98, '42tops': 0.99, 'max45heads': 0.98, 'max45tops': 0.99}


predictor = Predictor()

first = True


#def initialization():
    # IMG_78033622_good.JPEG
    # image_top1 = cv2.imread(
    #     '/home/alfred/Desktop/zippers/factory/tagged_images/all_tops_42/IMG_78033622_good.JPEG', 0)
    # image_head1 = cv2.imread(
    #     '/home/alfred/Desktop/zippers/factory/tagged_images/all_heads_42/IMG_2_goodhead.JPEG', 0)

    #image_top1 = cv2.imread(
    #    '/home/alfred/Desktop/zippers/factory/images/datasets/testheads1201CM/#IMG_12_testhead.JPEG', 0)
    #image_head1 = cv2.imread(
    #    '/home/alfred/Desktop/zippers/factory/images/datasets/test1201CM/IMG_12_test_1201CM_0.JPEG', 0)
    # image_top1 = reshape_input_image_for_cnn(image_top1, 'top')
    # image_head1 = reshape_input_image_for_cnn(image_head1, 'head')
    #print('true :' + str((image_top1.shape, image_head1.shape)))
    #prediction = predictor.predict(image_head1, image_top1, 'speed')
    # print(prediction)
    #return (prediction)


def show_img(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


@app.route("/")
def hello():
	print('hello from server')
	return 'OK'
    #return str(initialization())


im_counter = 0


@app.route('/images', methods=['GET', 'POST'])
def parse_request():
    global im_counter
    """ print(request.environ['QUERY_STRING'])"""
    qs = request.environ['QUERY_STRING']
    args = qs.split('&')
    zipper_type = args[0].split('=')[1]
    color = args[1].split('=')[1]
    size = args[2].split('=')[1]
    print((zipper_type, color, size))

    imag_1 = request.files['img_top'].read()
    imag_2 = request.files['img_head'].read()

    # imgarr_1 = np.asarray(bytearray(img_1), dtype=np.uint64)
    # imgarr_2 = np.asarray(bytearray(img_2), dtype=np.uint64 )

    imgarr_1 = np.fromstring(imag_1, np.uint8)
    imgarr_2 = np.fromstring(imag_2, np.uint8)

    # print((imgarr_1.shape, imgarr_2.shape))

    img_1 = cv2.imdecode(imgarr_1, 0)
    img_2 = cv2.imdecode(imgarr_2, 0)

    img_top = img_1
    img_head = img_2

    # cv2.imwrite('top' + str(im_counter) + '.jpg', img_top)
    # cv2.imwrite('head' + str(im_counter) + '.jpg', img_head)
    im_counter += 1

    # if (img_1.shape[1] == 226):
    #     img_top = img_1
    #     img_head = img_2
    # else:
    #     img_top = img_2
    #     img_head = img_1

    # image_top1 = reshape_input_image_for_cnn(img_top, 'top')
    # image_head1 = reshape_input_image_for_cnn(img_head, 'head')
    # print((image_top1.shape, image_head1.shape))
    # show_img(image_top1)
    # print(image_head1)

    prediction = predictor.predict(img_head, img_top, zipper_type)
    print(prediction)
    # cv2.imwrite(str(prediction) + '2.jpeg', img_top)
    # cv2.imwrite(str(prediction) + '.jpeg', img_head)
    return json.dumps({
        'status': 200,
        'label': prediction[2],
        'label_top': prediction[1],
        'label_head': prediction[0]
    })


#initialization()
