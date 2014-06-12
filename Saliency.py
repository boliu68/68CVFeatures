#Author: bliuab, the HKUST
#Date: 2013.12.30
#modifed from https://github.com/keremsahin1/MS_Thesis_Work
#/blob/ab024c0973aa98ed15b55b2788084d01e81bd8a8/fPeerGroupFiltering.m
from cv2 import *
import numpy as np
import math
import warnings

def PeerGroupFiltering(img, window_size):

    ouput_img = np.zeros(img.shape, np.uint8)

    height, width, bandwidth = img.shape

    k = window_size ** 2
    J = np.zeros([k, 1], np.float32)
    Dist = np.zeros([window_size, window_size], np.float32)
    Center = (window_size + 1) / 2
    window = np.zeros([window_size, window_size, bandwidth], np.float32)

    #Extend input img
    ext_img = extendImgbyMirror(img, window_size)

    gauss_kernel = matlab_style_gauss2d((window_size, window_size), 1)

    for i in range(Center - 1, height + Center - 1):
        for j in range(Center - 1, width + Center - 1):

            #copy current window
            for win_rowno in range(window_size):
                for win_colno in range(window_size):

                    window[win_rowno, win_colno] = ext_img[i - Center + 1 + win_rowno, j - Center + 1 + win_colno, :]

            #Find Euclidean distances between center pixel and other pixels
            for win_rowno in range(window_size):
                for win_colno in range(window_size):
                    Dist[win_rowno, win_colno] = math.sqrt(np.inner(window[Center - 1, Center - 1, :] - window[win_rowno, win_colno, :], window[Center - 1, Center - 1, :] - window[win_rowno, win_colno, :]))


            sort_dist = np.sort(Dist.reshape(Dist.shape[0] * Dist.shape[1]))
            sort_idx = np.argsort(Dist.reshape(Dist.shape[0] * Dist.shape[1]))

            #Fisher criterion
            for h in range(k):

                #Find a1
                a1 = sum(sort_dist[:h + 1]) / (h +1)

                #Find a2
                a2 = sum(sort_dist[h:k])

                if h != k:
                    a2 = a2 / (k - h)
                else:
                    a2 = 0
		
		#if a1 == 0:
		   # print 'a1'
		#if a2 == 0:
		    #print 'a2'

                #Find s1Sq
                s1Sq = sum((sort_dist[:(h+1)] - a1) ** 2)

                #Find s2Sq
                s2Sq = sum(((sort_dist[h:k]) - a2) ** 2)
		
		if s1Sq + s2Sq != 0:
		    J[h] = ((a1 - a2) ** 2) / (s1Sq + s2Sq)
		else:
		    J[h] = 0

            modified_gauss = np.array(gauss_kernel)
            groupsize = np.argmax(J)
            num_idx = sort_idx[(groupsize + 1):(k + 1)]
            row_idx, col_idx = num_idx / modified_gauss.shape[1], num_idx % modified_gauss.shape[1]
            modified_gauss[row_idx, col_idx] = 0
            #print i,j
            TotalWeights = np.sum(modified_gauss[:window_size, :window_size])
            WeightedTotal = np.array([np.sum(modified_gauss[:window_size, :window_size] * window[:window_size, :window_size, d]) for d in range(window.shape[2])])
            ouput_img[i - Center + 1, j - Center + 1, :] = WeightedTotal / TotalWeights

    return ouput_img

def extendImgbyMirror(img, window_size):

    ext_size = (window_size - 1) / 2
    height, width, bandwidth = img.shape

    output_img = np.zeros((height + 2 * window_size, width + 2 * window_size, bandwidth))
    output_img[ext_size:(height + ext_size), ext_size:(width + ext_size), :] = img

    for i in range(ext_size):
        output_img[i,:,:] = output_img[2 * ext_size - 1 - i,:,:]
        output_img[height + ext_size + i, :,:] = output_img[height + ext_size - 1 - i, :,:]

    for j in range(ext_size):
        output_img[:,j,:] = output_img[:, 2 * ext_size - 1 -j, :]
        output_img[:,width + ext_size + j, :] = output_img[:, width + ext_size - 1 - j, :]

    return output_img


def matlab_style_gauss2d(shape = (3,3), sigma = 1):

    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """

    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return hr

