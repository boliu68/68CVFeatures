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



def get_saliency(img):

    #calculate the contrast/ distance to the neighbor
    #k: the radius of neighbour.

    height, width = img.shape[:2]
    saliency_map = np.zeros([height, width], dtype = np.float32)
    radius = 1


    for h in range(height):
        for w in range(width):
            for k in range(radius):
                r = k + 1
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










