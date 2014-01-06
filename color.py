#Author: bliuab, the HKUST
#Date: 2013.12.20
from cv2 import *
from convert import *
import numpy as np
import math

def color(img):
    #mean of saturation
    #mean of hue
    #stadard deviation of saturation
    #standard deviation of hue

    hsv_img = cvtColor(img, COLOR_BGR2HSV)

    hue_mean = np.mean(hsv_img[:, :, 0])
    sat_mean = np.mean(hsv_img[:, :, 1])
    hue_std = np.std(hsv_img[:, :, 0])
    sat_std = np.std(hsv_img[:, :, 1])

    #Contrast of color
    b_var = np.var(img[:, :, 0])
    g_var = np.var(img[:, :, 1])
    r_var = np.var(img[:, :, 2])

    contrast_color = {'b':b_var, 'g':g_var, 'r':r_var}
    #print contrast_color

    #colorfulness
    colorfulness = get_colorfulness(img)

    #naturalness
    naturalness = get_naturalness(img)

    return [hue_mean, sat_mean, hue_std, sat_std, b_var, g_var, r_var, colorfulness, naturalness]

def get_colorfulness(img):

    #R - G
    rg = np.array(img[:, :, 2], np.int16)
    rg = rg - img[:, :, 1]

    #(R+G)/2 - B
    yb = np.array(img[:, :, 1], np.int16)
    yb = (yb + img[:, :, 2])/2 - img[:, :, 0]

    std_rgyb = math.sqrt(np.var(rg) + np.var(yb))
    mean_rgyb = math.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)

    colorful = std_rgyb + 0.3 * mean_rgyb

    return colorful

def get_naturalness(img):

    #Get natrualness
    #Pedro, J. S., & Siersdorfer, S. (2009).
    #Ranking and classifying attractiveness of photos in folksonomies.
    hsl_img = bgr2hsl(img)
    pixels = hsl_img[((hsl_img[:,:,2]<0.8) * (hsl_img[:,:,2]>0.2) * (hsl_img[:,:,1]> 0.1))]
    #print pixels.shape

    #print np.sum((hsl_img[:,:, 0]>=300))

    A = pixels[(pixels[:,0]>=25) * (pixels[:,0]<70)]
    num_A = np.sum((pixels[:, 0]>=25) * (pixels[:,0]<70))
    B = pixels[(pixels[:,0]>=95) * (pixels[:,0]<135)]
    num_B = np.sum((pixels[:,0]>=95) * (pixels[:,0]<135))
    C = pixels[(pixels[:,0]>=185) * (pixels[:,0]<260)]
    num_C = np.sum((pixels[:,0]>=185) * (pixels[:,0]<260))

    A_mean = np.mean(A[:,1])
    B_mean =  np.mean(B[:,1])
    C_mean = np.mean(C[:, 1])


    #print num_A, num_B, num_C
    N_skin = math.exp(-0.5 * ((np.mean(A[:,1]) - 0.76)/ 0.52) ** 2)
    N_grass = math.exp(-0.5 * ((np.mean(B[:,1]) - 0.81)/ 0.53) ** 2)
    N_sky = math.exp(-0.5 * ((np.mean(C[:, 1]) - 0.43)/ 0.22) ** 2)

    if math.isnan(N_skin):
        N_skin = 0
    if math.isnan(N_grass):
        N_grass = 0
    if math.isnan(N_sky):
        N_sky = 0

    N = (num_A * N_skin + num_B * N_grass + num_C * N_sky) / (num_A + num_B + num_C)

    return N
