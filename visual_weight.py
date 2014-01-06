#Author: bliuab, the HKUST
#Date: 2014.1.1
from cv2 import *
import numpy as np
import math

def visual_weight(img, subject_region):
    #The clarity contrast between subject region and the whole image

    gray = cvtColor(img, COLOR_BGR2GRAY)

    fimg = np.fft.fft2(gray)
    fshift_img = np.fft.fftshift(fimg)

    x0 = subject_region['x0']
    y0 = subject_region['y0']
    W = subject_region['W']
    H = subject_region['H']

    #print 'fimg shape', fshift_img.shape

    sb = gray[int(x0 - H / 2): int(x0 + H / 2), int(y0 - W / 2):int(y0 + W / 2)]
    fsb = np.fft.fft2(sb)
    fshift_sb = np.fft.fftshift(fsb)

    max_img = np.max(fshift_img)
    max_sb = np.max(fshift_sb)
    beta = 0.2

    # print max_img
    # print max_sb

    MI = np.sum(np.abs(fshift_img) > beta * max_img)
    MR = np.sum(np.abs(fshift_sb) > beta * max_sb)

    #print 'MI, MR', MI, MR

    fc = (MR * 1.00000 / (sb.shape[0] * sb.shape[1])) / (MI * 1.00000 / (gray.shape[0] * gray.shape[1]))

    return fc

