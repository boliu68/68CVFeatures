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

    #gray_img = np.divide(gray_img, 256.00000)

    mean = np.mean(yuv_img)
    var = np.var(yuv_img)
    #print mean
    #print var

    return [mean, var]