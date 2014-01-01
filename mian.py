#Author: bliuab, the HKUST
#Date: 2013.12.20
import cv2
from process import *


if __name__ == '__main__':

    np.set_printoptions(threshold=np.nan)

    #read and resize to 100 * 100
    img = cv2.imread('test.jpg', 1)
    img = cv2.resize(img, (100, 100))

    img_process(img)
    #imshow('img', img)
    #waitKey(0)



