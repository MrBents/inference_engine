from predictor import *
from skimage import io
import pandas as pd
import numpy as np
import cv2
import re

from os import listdir
from os.path import isfile, join


dirc = './2019-06-28T13:42:45-04:00/'

outdir = './outdata/'

def imread_convert(f):
    return cv2.imread(f)

files =  [f for f in listdir(dirc) if isfile(join(dirc, f))]

top, head = None, None

bound = len(files)

ctr = 0
for i in range(0, int(bound/2)):
    print(f'processing zipper {i}...', end='')
    for f_name in files:
        p = re.compile('[a-z]+_(b|g)(' + str(i) + ').[a-z]+')
        if p.match(f_name) != None:
            if 'top' in f_name:
                top = cv2.imread(join(dirc, f_name), 0)
            else:
                head = cv2.imread(join(dirc, f_name), 0)
            del files[files.index(f_name)]
    pred = predictor.predict(head, top, '45')

    lbl = {"label" : pred[2]}
    
    path_top, path_head = '', ''

    if lbl["label"] == 0:
        path_top, path_head = './outdata/top_b' + str(ctr) + '.jpg', './outdata/head_b' + str(ctr) + '.jpg'
        cv2.imwrite(path_head, head)
        cv2.imwrite(path_top, top)
    else:
        path_top, path_head = './outdata/top_g' + str(ctr) + '.jpg', './outdata/head_g' + str(ctr) + '.jpg'
        cv2.imwrite(path_head, head)
        cv2.imwrite(path_top, top)
    ctr += 1
    print('Done')


            

