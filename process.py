#Author: bliuab, the HKUST
#Date: 2013.12.20
from cv2 import *
import numpy as np
from lighting import *
from color import *
from sharpness import *

def img_process(img):

    #lighting 2 features
    mean, std = lighting(img)

    #colors
    color(img)

    #sharpness blur
    sharpness = sharpness_blur(img)
    print sharpness


