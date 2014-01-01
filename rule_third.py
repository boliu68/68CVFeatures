#Author: bliuab, the HKUST
#Date: 2014.1.1
from cv2 import *
import numpy as np
import math


def rule_of_third(img, subject_region):

    height, width = img.shape[:2]

    x0 = subject_region['x0']
    y0 = subject_region['y0']

    stress_point = [[height / 3, width / 3], [height / 3, width / 3 * 2], [height / 3 * 2, width / 3], [height / 3 * 2, width / 3 * 2]]

    min_dist = np.min([math.sqrt((x0 - stress_point[i][0]) ** 2 + (y0 - stress_point[i][1]) ** 2) for i in range(4)])

    #print [math.sqrt((x0 - stress_point[i][0]) ** 2 + (y0 - stress_point[i][1]) ** 2) for i in range(4)]

    return min_dist
