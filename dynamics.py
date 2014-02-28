#Author: bliuab, the HKUST
#Date: 2014.1.2
from cv2 import *
import numpy as np
import math

def get_dynamics(img):

    gray = cvtColor(img, COLOR_BGR2GRAY)
    edges = Canny(gray, 50, 200)
    lines = lines = HoughLinesP(edges, 1, np.pi/180, 1, minLineLength = 5 )

    lines = np.array(lines[0])
    x0 = lines[:, 0]
    y0 = lines[:, 1]
    x1 = lines[:, 2]
    y1 = lines[:, 3]

    length = np.sqrt(((x0 - x1) ** 2) + ((y0 - y1) ** 2))
    #
    # for i in range(len(x0)):
    #     line(img, (x0[i], y0[i]), (x1[i], y1[i]), (0, 255, 0), 2)
    #
    # imshow('lines', img)
    # waitKey(0)

    theta = np.arctan2((y1 - y0) * 1.0000, (x1 - x0) * 1.0000)

    stat_idx = (theta < - 75 * math.pi / 180) + (theta > 75 * math.pi / 180) + ((theta > (-15 * math.pi / 180)) * (theta < (15 * math.pi / 180)))
    dyna_idx = True - stat_idx

    num_stat = np.sum(stat_idx)
    num_dyna = np.sum(dyna_idx)

    len_stat = np.sum(length[stat_idx])
    len_dyna = np.sum(length[dyna_idx])

    if num_dyna == 0:
        num_dyna = 1
    if len_dyna == 0:
        len_dyna = 1

    return [num_stat * 1.000 / num_dyna, len_stat * 1.000 / len_dyna]

