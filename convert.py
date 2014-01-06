#Author: bliuab, the HKUST
#Date: 2013.12.20
from cv2 import *
import numpy as np
import math


def bgr2hsl(img):

    #convert the brg space to hsl space
    bgr_ = np.array(img, np.float32)
    bgr_ = np.divide(bgr_, 255.0)

    #print bgr_
    Cmax = np.amax(bgr_, 2)
    Cmin = np.amin(bgr_, 2)
    Cmax_ind = np.argmax(bgr_, 2)
    Cmin_ind = np.argmin(bgr_, 2)

    delta = Cmax - Cmin
    delta_nonzero = delta + np.ones(delta.shape) * (delta == 0)

    height, width = img.shape[:2]
    H = np.zeros([height, width], np.float16)
    S = np.zeros([height, width], np.float16)
    L = np.zeros([height, width], np.float16)

    H = (60 * (np.mod((bgr_[:, :, 1] - bgr_[:, :, 0])*1.00/ delta_nonzero, 6))) * ((Cmax_ind == 2) * (delta != 0))
    H = H + ((60 * ((bgr_[:, :, 0] - bgr_[:, :, 2])* 1.00/ delta_nonzero + 2)) * ((Cmax_ind == 1) * (delta != 0)))
    H = H + ((60 * ((bgr_[:, :, 2] - bgr_[:, :, 1])* 1.00 / delta_nonzero + 4)) * ((Cmax_ind == 0) * (delta != 0)))


    L = (Cmax + Cmin) / 2

    try:
        S = (delta / (1 - np.absolute(2 * L - 1))) * (delta != 0)
    except:
        pass

    #print img[0,0,:]
    #print H[0,0], L[0,0], S[0,0]
    hsl = np.array(img, np.float16)
    hsl[:, :, 0] = H
    hsl[:, :, 1] = S
    hsl[:, :, 2] = L

    S[np.isnan(S)] = 1
    #print hsl.shape

    return hsl