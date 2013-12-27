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
    img3 = np.array(img * saliency_map3, dtype = np.float32)


    sb_lgt_mean, sb_lgt_var = lighting(img3)
    [sb_hue_mean, sb_sat_mean, sb_hue_std, sb_sat_std, sb_contrast_color, sb_colorfulness, sb_naturalness] = color(img3)
    sb_sharpness = sharpness_blur(img3)

    return [sb_lgt_mean, sb_lgt_var, sb_hue_mean, sb_sat_mean, sb_hue_std, sb_sat_std, sb_contrast_color, sb_colorfulness, sb_naturalness, sb_sharpness]


def get_saliency(img):

    #calculate the contrast/ distance to the neighbor
    #k: the radius of neighbour.

    height, width = img.shape[:2]
    saliency_map = np.zeros([height, width], dtype = np.float16)
    radius = 1


    for h in range(height):
        for w in range(width):
            for k in range(radius):
                r = k + 1
                #print h, w
                neighbor = np.array(img[h, w, :])
                if h - r >= 0:
                    if w - r >= 0:
                        #saliency_map[h, w] += math.sqrt((img[h, w, :] - img[h-r, w-r, :]) ** 2)
                        neighbor = np.vstack([neighbor, img[h - r, w - r, :]])
                    if w + r < width:
                        neighbor = np.vstack([neighbor, img[h - r, w + r, :]])
                    neighbor = np.vstack([neighbor, img[h - r, w, :]])

                if h + r < height:
                    if w - r >= 0:
                        #saliency_map[h, w] += math.sqrt((img[h, w, :] - img[h-r, w-r, :]) ** 2)
                        neighbor = np.vstack([neighbor, img[h + r, w - r, :]])
                    if w + r < width:
                        neighbor = np.vstack([neighbor, img[h + r, w + r, :]])
                    neighbor = np.vstack([neighbor, img[h + r, w, :]])

                if w - r >= 0:
                    neighbor = np.vstack([neighbor, img[h, w - r, :]])

                if w + r < width:
                    neighbor = np.vstack([neighbor, img[h, w + r, :]])

                saliency_map[h, w] += math.sqrt(sum([sci.inner(img[h, w, :] - neigh, img[h, w, :] - neigh) for neigh in neighbor]))

    #normalize
    saliency_map = (saliency_map / (np.max(saliency_map) - np.min(saliency_map)))

    #imshow('saliency', saliency_map)
    #waitKey(0)
    return saliency_map










