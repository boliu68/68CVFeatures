#Author: bliuab, the HKUST
#Date: 2013.12.20
from cv2 import *
import numpy as np
from lighting import *
from color import *
from sharpness import *
from subject import *

def img_process(img):

    #lighting 2 features
    lgt_mean, lgt_var = lighting(img)

    #colors
    [hue_mean, sat_mean, hue_std, sat_std, contrast_color, colorfulness, naturalness] = color(img)

    #sharpness blur
    sharpness = sharpness_blur(img)

    #subject quality
    [subject_region, sb_lgt_mean, sb_lgt_var, sb_hue_mean, sb_sat_mean, sb_hue_std, sb_sat_std, sb_contrast_color, sb_colorfulness, sb_naturalness, sb_sharpness] = subject(img)

    