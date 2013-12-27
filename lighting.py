#Author: bliuab, the HKUST
#Date: 2013.12.20
from cv2 import *
import numpy as np

def lighting(img):
    #return normalized 0-1
    #pixel intensity in gray's mean
    #pixel intensity in gray's variance
    #2 features
    gray_img = cvtColor(img, COLOR_BGR2GRAY)
    yuv_img = cvtColor(img, COLOR_BGR2YUV)

    y_img = yuv_img[:, :, 0]

    mean = np.mean(y_img)
    var = np.var(y_img)

    return [mean, var]