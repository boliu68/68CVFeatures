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
from color_emotion import *
import time

def img_process(img, writer, path, label):

    start = float(time.clock())

    fea = {}
    fea_vec = [path, label]

    #lighting 2 features
    lgt_mean, lgt_var = lighting(img, None)
    fea['lighting_fea'] = [lgt_mean, lgt_var ]
    fea_vec += [lgt_mean, lgt_var]
    #colors
    [hue_mean, sat_mean, hue_std, sat_std, b_var, g_var, r_var, colorfulness, naturalness] = color(img, None)
    fea['color_fea'] = [hue_mean, sat_mean, hue_std, sat_std, b_var, g_var, r_var, colorfulness, naturalness]
    fea_vec += [hue_mean, sat_mean, hue_std, sat_std, b_var, g_var, r_var, colorfulness, naturalness]
    
    time1 = float(time.clock())
    print "Light Time:", time1 - start
    
    #sharpness blur
    sharpness = sharpness_blur(img)
    fea['sharpness_fea'] = sharpness
    fea_vec += [sharpness['Sh'], sharpness['Sh_std'], sharpness['blur']]

    time2 = float(time.clock())
    print "Sharpness:", time2 - time1

    #subject  quality
    [saliency_map, subject_region, sb_lgt_mean, sb_lgt_var, sb_hue_mean, sb_sat_mean, sb_hue_std, sb_sat_std, sb_bvar, sb_gvar, sb_rvar, sb_colorfulness, sb_naturalness, sb_sharpness] = subject(img)
    fea['subject_fea'] = [sb_lgt_mean, sb_lgt_var, sb_hue_mean, sb_sat_mean, sb_hue_std, sb_sat_std, sb_bvar, sb_gvar, sb_rvar, sb_colorfulness, sb_naturalness, sb_sharpness]
    fea_vec += [sb_lgt_mean, sb_lgt_var, sb_hue_mean, sb_sat_mean, sb_hue_std, sb_sat_std, sb_bvar, sb_gvar, sb_rvar, sb_colorfulness, sb_naturalness, sb_sharpness['Sh'], sb_sharpness['Sh_std'], sb_sharpness['blur']]
    #[saliency_map, subject_region] = subject(img)
    
    time3 = float(time.clock())
    print "Subject:", time3 - time2

    #rule of third
    rule3rd = rule_of_third(img, subject_region)
    fea['rule3rd_fea'] = rule3rd
    fea_vec += [rule3rd]
    time4 = float(time.clock())
    print "Rule of third:", time4 - time3

    #obtain the simplicity features
    [qimg, qimg3, qh, qh_sub] = get_simplicity(img, saliency_map, subject_region)
    fea['simplicity_fea'] = [qimg, qimg3, qh, qh_sub]
    fea_vec += [qimg, qimg3, qh, qh_sub]

    time5 = float(time.clock())
    print "Simplicity:", time5 - time4

    #to get the clarity contrast between subject region and original image
    fc = visual_weight(img, subject_region)
    fea['vis_weight_fea'] = [fc]
    fea_vec += [fc]
    time6 = float(time.clock())
    print "Visual wegiht:", time6 - time5

    #to get the ratio of number and length of static and dynamic lines
    ratio_num, ratio_len = get_dynamics(img)
    fea['dynamic_fea'] = [ratio_num, ratio_len]
    fea_vec += [ratio_num, ratio_len]
    #to obtain the color histogram for emotion
    hist = get_colemotion(img)
    fea['color_hist_fea'] = hist
    fea_vec += hist
    time7 = float(time.clock())
    print "Dynamic:", time7 - time6

    writer.writerow(fea_vec)

    return fea
