#Author: bliuab, the HKUST
#Date: 2013.12.24
from cv2 import *
import numpy as np
from lighting import *
from color import *
from sharpness import *
import scipy.cluster.vq as cluster_vq
from scipy.cluster.vq import vq
from Saliency import *
from attend_view import *

def subject(img):

    #gray_img = CvtColor(img, COLOR_BGR2GRAY)

    saliency_map = get_saliency(img)
    
    [x0, y0, W, H] = attend_view(saliency_map) #get the subject area

    saliency_map3 = np.zeros((saliency_map.shape[0], saliency_map.shape[1], 3))

    sb_lgt_mean, sb_lgt_var = lighting(img, saliency_map)
    [sb_hue_mean, sb_sat_mean, sb_hue_std, sb_sat_std, sb_bvar, sb_gvar, sb_rvar, sb_colorfulness, sb_naturalness] = color(img, saliency_map)
    sb_sharpness = sharpness_blur(img[int(x0 - H / 2):int(x0 + H / 2), int(y0 - W / 2):int(y0 + W / 2), :])

    return [saliency_map, {'x0':x0, 'y0':y0, 'W':W, 'H':H}, sb_lgt_mean, sb_lgt_var, sb_hue_mean, sb_sat_mean, sb_hue_std, sb_sat_std, sb_bvar, sb_gvar, sb_rvar, sb_colorfulness, sb_naturalness, sb_sharpness]


def get_saliency(img):

    #calculate the contrast/ distance to the neighbor
    #give the radius of neighbour.
    radius = 1

    #change color space and color quantization
    luv = cvtColor(img, COLOR_BGR2LUV)
    
    #preprocessing
    #luv = PeerGroupFiltering(luv, 3)
    #luv = block_divide(luv, 2)

    #generate the saliency map
    saliency_map = saliency_distance(luv, radius)

    #gaussian smooth and normalize saliency
    saliency_map = medianSmooth(saliency_map, 3)
    saliency_norm = ((saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map)))
    
    return saliency_norm

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
    attended_area = attended_area * 255

    return attended_area

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

            left = w - radius
            right = w + radius
            top = h - radius
            bottom = h + radius

            if left < 0:
                left = 0
            if right > width:
                right = width
            if top < 0:
                top = 0
            if bottom > height:
                bottom = height

            neighbor = luv[top:bottom, left:right, :]
            neighbor = np.reshape(neighbor, (neighbor.shape[0] * neighbor.shape[1], 3))

            saliency_map[h, w] += sum([gaussian_distance(luv[h, w, :], neigh) for neigh in neighbor])

    return saliency_map


def gaussian_distance(x, y):

    elu_dist = math.sqrt(np.dot((x - y), (x-y).transpose()))
    return elu_dist

def color_quant(img, k):
    #color quantization using kmeans

    pixel = np.reshape(img, (img.shape[0] * img.shape[1], 3))
    centroids, _ = cluster_vq.kmeans(pixel, k)
    qnt, _ = vq(pixel, centroids)

    centers_idx = np.reshape(qnt, (img.shape[0], img.shape[1]))
    clustered = centroids[centers_idx]

    return clustered

def block_divide(img, n):

    height = img.shape[0]
    width = img.shape[1]

    di_img = np.array(img)

    for i in range(height / n):
       for j in range(width / n):
           di_img[i * n : (i + 1) * n, j * n : (j + 1) * n, 0] = np.mean(di_img[i * n : (i + 1) * n, j * n : (j + 1) * n, 0])
           di_img[i * n : (i + 1) * n, j * n : (j + 1) * n, 1] = np.mean(di_img[i * n : (i + 1) * n, j * n : (j + 1) * n, 1])
           di_img[i * n : (i + 1) * n, j * n : (j + 1) * n, 2] = np.mean(di_img[i * n : (i + 1) * n, j * n : (j + 1) * n, 2])

       di_img[i * n : (i + 1) * n, (j + 1) * n::, 0] = np.mean(di_img[i * n : (i + 1) * n, (j + 1) * n::, 0])
       di_img[i * n : (i + 1) * n, (j + 1) * n::, 1] = np.mean(di_img[i * n : (i + 1) * n, (j + 1) * n::, 1])
       di_img[i * n : (i + 1) * n, (j + 1) * n::, 2] = np.mean(di_img[i * n : (i + 1) * n, (j + 1) * n::, 2])

    for j in range(width / n):
       di_img[(i + 1) * n::, j * n : (j + 1) * n, 0] = np.mean(di_img[(i + 1) * n::, j * n : (j + 1) * n, 0])
       di_img[(i + 1) * n::, j * n : (j + 1) * n, 1] = np.mean(di_img[(i + 1) * n::, j * n : (j + 1) * n, 1])
       di_img[(i + 1) * n::, j * n : (j + 1) * n, 2] = np.mean(di_img[(i + 1) * n::, j * n : (j + 1) * n, 2])


    return di_img

def medianSmooth(saliency, radius):

    height, width = saliency.shape[:2]
    smooth_saliency = np.array(saliency)

    for h in range(height):
        for w in range(width):

            left = w - radius
            right = w + radius
            top = h - radius
            bottom = h + radius

            if left < 0:
                left = 0
            if right >= width:
                right = width - 1
            if top < 0:
                top = 0
            if bottom >= height:
                bottom = height

            neighbor = saliency[top:bottom, left:right]
            #neighbor = np.reshape(neighbor, (neighbor.shape[0] * neighbor.shape[1], 1))

            smooth_saliency[h, w] = np.mean(neighbor)

    return smooth_saliency
