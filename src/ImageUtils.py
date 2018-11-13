import os
from openface import data
import cv2
import numpy as np

fileDir = os.path.dirname(os.path.realpath(__file__))

def decode_image_buffer(image_buffer):
    return cv2.imdecode(np.array(image_buffer), cv2.IMREAD_COLOR)

def rgb_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def grayscale_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

def normalize(image):
    normalized_image = np.zeros((800, 800))
    return cv2.normalize(image,  normalized_image, 0, 255, cv2.NORM_MINMAX)