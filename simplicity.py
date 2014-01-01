#Author: bliuab, the HKUST
#Date: 2014.1.1
from cv2 import *
import numpy as np
import math


def get_simplicity(img, saliency_map, subject_region):

    #edge distribution of origin imgage
    #and saliency reweighted image

    qimg = edge_distr(img)

    #saliency reweighted image
    saliency_map3 = np.zeros((saliency_map.shape[0], saliency_map.shape[1], 3))

    saliency_map3[:, :, 0] = saliency_map
    saliency_map3[:, :, 1] = saliency_map
    saliency_map3[:, :, 2] = saliency_map
    img3 = np.array(img * saliency_map3, dtype=np.float32)

    qimg3 = edge_distr(img3)

    #distinct hue count of origin image
    #and subject region

    qh = hue_count(img)


    x0 = subject_region['x0']
    y0 = subject_region['y0']
    W = subject_region['W']
    H = subject_region['H']

    qh_sub = hue_count(img[int(x0 - H /2):int(x0 + H / 2), int(y0 - W / 2):int(y0 + W / 2), :])

    return [qimg, qimg3, qh, qh_sub]

def edge_distr(img):

    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]

    kernel_size = 3
    ddepth = CV_32F

    laplac_b = Laplacian(b, ddepth, kernel_size)
    laplac_g = Laplacian(g, ddepth, kernel_size)
    laplac_r = Laplacian(r, ddepth, kernel_size)

    laplac_img = np.abs((laplac_b + laplac_g + laplac_r) / 3)

    #if laplac_img.shape != (100, 100):

    #normalize to sum to 1
    laplac_img = laplac_img / np.sum(laplac_img)

    # imshow('laplac', laplac_img)
    # waitKey(0)

    Px = np.sum(laplac_img, 1)
    Py = np.sum(laplac_img, 0)

    wx = get_width(Px, 0.98)
    wy = get_width(Py, 0.98)

    q = 1 - wx * wy
    return q

def get_width(P, threshold):

    P.sort()
    #print P
    for i in range(P.shape[0]):
        idx = P.shape[0] - i - 1
        if np.sum(P[idx::]) >= threshold:
            return ((i + 1) * 1.0000 / P.shape[0])

def hue_count(img):

    hsv = cvtColor(img, COLOR_BGR2HSV)

    good_id = (hsv[:,:,1] > 0.2 * 255) * (hsv[:,:,2] > 0.15 * 255) * (hsv[:,:,2] < 0.95 * 255)

    good_h = hsv[good_id, 0]

    bin, _ = np.histogram(good_h, 20)

    m = np.max(bin)
    a = 0.05
    N = np.sum(bin > a * m)

    qh = 20 - N


    return qh
