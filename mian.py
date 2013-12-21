#Author: bliuab, the HKUST
#Date: 2013.12.20
import cv2
from process import *


if __name__ == '__main__':

    np.set_printoptions(threshold=np.nan)

    img = cv2.imread('test.jpg', 1)
    img_process(img)
    #imshow('img', img)
    #waitKey(0)



