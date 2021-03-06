#Author: bliuab, the HKUST
#Date: 2013.12.20
from cv2 import *
from convert import *
import numpy as np
import math

def sharpness_blur(img):

    #Laplacian image
    kernel_size = 3
    ddepth = CV_32F
    gray_img = cvtColor(img, COLOR_BGR2GRAY)
    yuv_img = cvtColor(img, COLOR_BGR2YUV)

    laplac_img = Laplacian(gray_img, ddepth, kernel_size)
    near_avg = avg_lum(yuv_img)

    Sh = np.mean(laplac_img / near_avg)
    Sh_std = np.std(laplac_img / near_avg)

    #print Sh
    #print Sh_std

    #FFT transformation of image
    f = np.fft.fft2(gray_img)
    fshift = np.fft.fftshift(f)

    blur = np.sum(fshift > 5) * 1.0000/ (gray_img.shape[0] * gray_img.shape[1])
    #print blur

    sharpness = {'Sh':Sh, 'Sh_std': Sh_std, 'blur':blur}

    return sharpness


def avg_lum(yuv_img):
    #the averge luminance

    lum_img = yuv_img[:, :, 0]
    h, w = lum_img.shape[:2]
    avg = np.zeros(lum_img.shape, np.float16)

    for i in range(h):
        for j in range(w):
            left = i - 1
            top = j - 1
            right = i + 1
            bottom = j + 1

            if i - 1 < 0:
                left = 0
            if j - 1 < 0:
                top = 0
            if i + 1 > h:
                right = h
            if j + 1 > w:
                bottom = w

            avg[i, j] = np.mean(lum_img[left: right, top : bottom])

            if avg[i, j] == 0:
                avg[i, j] = 1

    return avg

