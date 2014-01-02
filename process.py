#Author: bliuab, the HKUST
#Date: 2013.12.20
from cv2 import *
import numpy as np
from lighting import *
from color import *
from sharpness import *
from subject import *
from rule_third import *
from simplicity import *
from visual_weight import *
from dynamics import *

def img_process(img):

    # #lighting 2 features
    # lgt_mean, lgt_var = lighting(img)
    #
    # #colors
    # [hue_mean, sat_mean, hue_std, sat_std, contrast_color, colorfulness, naturalness] = color(img)
    #
    # #sharpness blur
    # sharpness = sharpness_blur(img)
    #
    # #subject quality
    # [saliency_map, subject_region, sb_lgt_mean, sb_lgt_var, sb_hue_mean, sb_sat_mean, sb_hue_std, sb_sat_std, sb_contrast_color, sb_colorfulness, sb_naturalness, sb_sharpness] = subject(img)
    #
    # #rule of third
    # rule3rd = rule_of_third(img, subject_region)
    #
    # #obtain the simplicity features
    # [qimg, qimg3, qh, qh_sub] = get_simplicity(img, saliency_map, subject_region)
    #
    # #print qh, qh_sub
    #
    # #to get the clarity contrast between subject region and original image
    # fc = visual_weight(img, subject_region)

    ratio_num, ratio_len = get_dynamics(img)

    print ratio_num, ratio_len
