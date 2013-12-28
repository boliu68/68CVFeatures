#Author: bliuab, the HKUST
#Date: 2013.12.24
from cv2 import *
import numpy as np
from lighting import *
from color import *
from sharpness import *
import scipy.spatial.distance as sci_dist
import scipy as sci

def subject(img):

    #gray_img = CvtColor(img, COLOR_BGR2GRAY)

    saliency_map = get_saliency(img)

    saliency_map3 = np.zeros((saliency_map.shape[0], saliency_map.shape[1], 3))
    saliency_map3[:, :, 0] = saliency_map
    saliency_map3[:, :, 1] = saliency_map
    saliency_map3[:, :, 2] = saliency_map
    img3 = np.array(img * saliency_map3, dtype=np.float32)


    sb_lgt_mean, sb_lgt_var = lighting(img3)
    [sb_hue_mean, sb_sat_mean, sb_hue_std, sb_sat_std, sb_contrast_color, sb_colorfulness, sb_naturalness] = color(img3)
    sb_sharpness = sharpness_blur(img3)

    return [sb_lgt_mean, sb_lgt_var, sb_hue_mean, sb_sat_mean, sb_hue_std, sb_sat_std, sb_contrast_color, sb_colorfulness, sb_naturalness, sb_sharpness]


def get_saliency(img):

    #calculate the contrast/ distance to the neighbor
    #give the radius of neighbour.
    block_img = np.array(img)
    radius = 1
    #n = 3

    #imshow('block_img', block_img)
    #waitKey(0)

    luv = cvtColor(block_img, COLOR_BGR2LUV)

    #imshow('luv', luv)
    #waitKey(0)

    #generate the saliency map
    saliency_map = saliency_distance(luv, radius)

    #normalize
    saliency_map = (saliency_map / (np.max(saliency_map) - np.min(saliency_map)))

    #imshow('saliency', saliency_map)
    #waitKey(0)

    fuzzy_grow(saliency_map)

    return saliency_map

def fuzzy_grow(saliency_map):

    attended_area = np.ones(saliency_map.shape, dtype=np.uint8)
    gk = 0.5 #partition threshold

    #print sum(sum(saliency_map > gk))
    hk = [sum(sum(saliency_map > gk)), sum(sum(saliency_map <= gk))]

    p_gk = [hk_i * 1.00000 / sum(hk) for hk_i in hk]
    min_Tau = np.inf
    #print p_gk
    for a in range(10):
        for u in range(a):
            Tau = optim_ua(p_gk, gk, a * 1.0 / 10, u * 1.0 / 10)

            if np.isinf(min_Tau) or Tau < min_Tau:
                min_Tau = Tau
                min_a = a * 1.00 / 10
                min_u = u * 1.00 / 10

    #fuzzy growing the saliency map
    seed = np.max(saliency_map)
    s = (min_a + min_u) / 2


    attended_area = attended_area * ((saliency_map < seed) * (saliency_map > s))

    #print s

    attended_area = attended_area * 255

    imshow('grow map', attended_area)
    waitKey(0)

def optim_ua(p_gk, gk, a, u):

    #c-patition cross entropy
    mu_a = 0
    mu_b = 0

    if gk >= a:
        mu_a = 1
        mu_b = 0
    elif gk < a and gk > u:
        mu_a = (gk - u) / (a - u)
        mu_b = (gk - a) / (u - a)
    elif gk <= u:
        mu_a = 0
        mu_b = 1

    P_Ba = sum([mu_a * p_gk0 for p_gk0 in p_gk])
    P_Bu = sum([mu_b * p_gk0 for p_gk0 in p_gk])

    H_A = - p_gk[0] / P_Ba * math.log(p_gk[0] / P_Ba) - p_gk[1] / P_Ba * math.log(p_gk[1] / P_Ba)
    H_U = - p_gk[0] / P_Bu * math.log(p_gk[0] / P_Bu) - p_gk[1] / P_Bu * math.log(p_gk[1] / P_Bu)

    Tau = (H_A - H_U) ** 2

    return Tau

def saliency_distance(luv, radius):

    saliency_map = np.zeros(luv.shape[:2])
    height = luv.shape[0]
    width = luv.shape[1]

    for h in range(height):
        for w in range(width):
            for k in range(radius):
                r = k + 1
                #print h, w
                neighbor = np.array(luv[h, w, :])
                if h - r >= 0:
                    if w - r >= 0:
                        #saliency_map[h, w] += math.sqrt((img[h, w, :] - img[h-r, w-r, :]) ** 2)
                        neighbor = np.vstack([neighbor, luv[h - r, w - r, :]])
                    if w + r < width:
                        neighbor = np.vstack([neighbor, luv[h - r, w + r, :]])
                    neighbor = np.vstack([neighbor, luv[h - r, w, :]])

                if h + r < height:
                    if w - r >= 0:
                        #saliency_map[h, w] += math.sqrt((img[h, w, :] - img[h-r, w-r, :]) ** 2)
                        neighbor = np.vstack([neighbor, luv[h + r, w - r, :]])
                    if w + r < width:
                        neighbor = np.vstack([neighbor, luv[h + r, w + r, :]])
                    neighbor = np.vstack([neighbor, luv[h + r, w, :]])

                if w - r >= 0:
                    neighbor = np.vstack([neighbor, luv[h, w - r, :]])

                if w + r < width:
                    neighbor = np.vstack([neighbor, luv[h, w + r, :]])

                saliency_map[h, w] += sum([math.sqrt(sci.inner(luv[h, w, :] - neigh, luv[h, w, :] - neigh)) for neigh in neighbor])

    return saliency_map


def preprocess(img):
     #for i in range(height / n):
    #    for j in range(width / n):
    #        block_img[i * n : (i + 1) * n, j * n : (j + 1) * n, 0] = np.mean(block_img[i * n : (i + 1) * n, j * n : (j + 1) * n, 0])
    #        block_img[i * n : (i + 1) * n, j * n : (j + 1) * n, 1] = np.mean(block_img[i * n : (i + 1) * n, j * n : (j + 1) * n, 1])
    #        block_img[i * n : (i + 1) * n, j * n : (j + 1) * n, 2] = np.mean(block_img[i * n : (i + 1) * n, j * n : (j + 1) * n, 2])
    #
    #    block_img[i * n : (i + 1) * n, (j + 1) * n::, 0] = np.mean(block_img[i * n : (i + 1) * n, (j + 1) * n::, 0])
    #    block_img[i * n : (i + 1) * n, (j + 1) * n::, 1] = np.mean(block_img[i * n : (i + 1) * n, (j + 1) * n::, 1])
    #    block_img[i * n : (i + 1) * n, (j + 1) * n::, 2] = np.mean(block_img[i * n : (i + 1) * n, (j + 1) * n::, 2])
    #
    #for j in range(width / n):
    #    block_img[(i + 1) * n::, j * n : (j + 1) * n, 0] = np.mean(block_img[(i + 1) * n::, j * n : (j + 1) * n, 0])
    #    block_img[(i + 1) * n::, j * n : (j + 1) * n, 1] = np.mean(block_img[(i + 1) * n::, j * n : (j + 1) * n, 1])
    #    block_img[(i + 1) * n::, j * n : (j + 1) * n, 2] = np.mean(block_img[(i + 1) * n::, j * n : (j + 1) * n, 2])
    pass


