#Author: bliuab, the HKUST
#Date: 2013.12.20
import cv2
import os
from process import *
import csv

if __name__ == '__main__':


	img_dir = 'data/data/images'
	label_path = 'data/AllUser.label.txt'
	np.set_printoptions(threshold=np.nan)
	
	output_filename = 'image_features.csv'
	output_file = open(output_filename, 'wb')
	writer = csv.writer(output_file)
	
	label_idx = (open(label_path, 'r')).readlines()
	

	for i in range(len(label_idx)):
		info = label_idx[i].split(',')
		label = int(info[1])
		img_idx = info[0]
		print 'label', label, 'image', img_idx 
		img_path = os.path.join(img_dir, img_idx + '.jpg')
		image = cv2.imread(img_path, 1)
		resize_img = cv2.resize(image, (100, 100))
		fea = img_process(resize_img, writer, img_idx, label)
		output_file.flush()
		
	output_file.close()

    #read and resize to 100 * 100
    #img = cv2.imread('test.jpg', 1)
    #img = cv2.resize(img, (100, 100))
    #
    #fea = img_process(img)
    #imshow('img', img)
    #waitKey(0)

