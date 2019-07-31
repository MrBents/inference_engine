from time import sleep
import os
import numpy as np
import cv2
#from flask import Flask, request
import requests


def post_images(img_file):
    """ post image and return the response """
    img = open(img_file, 'rb').read()
    response = requests.post(URL, data=img, headers=headers)
    return response


image_names_heads = os.listdir(
    '/home/alfred/Desktop/zippers/factory/tagged_images/other_new_heads/')

image_names_tops = os.listdir(
    '/home/alfred/Desktop/zippers/factory/tagged_images/other_new_heads/')


lista = []
for i in range(100, 110):
    lista.append(
        '/home/alfred/Desktop/zippers/factory/tagged_images/other_new_heads/' + image_names_heads[i])

bol = True
while bol:
    for i in lista:
        post_image(i)
        sleep(0.1)
