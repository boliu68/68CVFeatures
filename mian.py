#Author: bliuab, the HKUST
#Date: 2013.12.20
import cv2
import os
from process import *
import csv

if __name__ == '__main__':

    np.set_printoptions(threshold=np.nan)

    cate_files = os.listdir('SelectedExamples')
    output_filename = 'image_features.csv'
    output_file = open(output_filename, 'wb')
    writer = csv.writer(output_file)

    for cate in cate_files:
        if 'Top' in cate:
            label = 1
        else:
            label = 0

        img_dir = os.path.join('SelectedExamples', cate)
        imgs = os.listdir(img_dir)

        for path in imgs:
            img_path = os.path.join(img_dir, path)
            image = cv2.imread(img_path, 1)
            resize_img = cv2.resize(image, (100, 100))
            fea = img_process(resize_img, writer, path, label)
            output_file.flush()

    output_file.close()

    #read and resize to 100 * 100
    #img = cv2.imread('test.jpg', 1)
    #img = cv2.resize(img, (100, 100))
    #
    #fea = img_process(img)
    #imshow('img', img)
    #waitKey(0)

