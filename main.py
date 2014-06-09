#Author: bliuab, the HKUST
#Date: 2013.12.20
import cv2
import os
from process import *
import csv
import numpy as np
import math


def get_fea(start, end, index, fail_list):


	img_dir = 'data/data/images'
	label_path = 'data/AllUser.label.txt'
	np.set_printoptions(threshold=np.nan)
	
	output_filename = '2image_features_%d.csv' % index
	output_file = open(output_filename, 'wb')
	writer = csv.writer(output_file)
	
	label_idx = (open(label_path, 'r')).readlines()

	run = True;

	for i in range(len(label_idx)):
	    if i >= start and i < end:	
		info = label_idx[i].split(',')
		label = int(info[1])
		img_idx = info[0]

		print 'index:', index, ',label', label, ',image', img_idx 
		print '-------------'
		img_path = os.path.join(img_dir, img_idx + '.jpg')
		image = cv2.imread(img_path, 1)
		
		if image != None:
		    resize_img = cv2.resize(image, (100, 100))
		    fea = img_process(resize_img, writer, img_idx, label)
		    output_file.flush()
		    if fea == False:
			fail_list.write('%s \n' % img_idx)
		else:
		    fail_list.write('%s \n' % img_idx)
		
		fail_list.flush()
	
	output_file.close()

    #read and resize to 100 * 100
    #img = cv2.imread('test.jpg', 1)
    #img = cv2.resize(img, (100, 100))
    #
    #fea = img_process(img)
    #imshow('img', img)
    #waitKey(0)

if __name__ == '__main__':
	
	fail_list = open('Fail_list.txt', 'w+')
	label_path = 'data/AllUser.label.txt'
	label_idx = (open(label_path, 'r')).readlines()
	thread_num = 15
	zero = 0 
	
	step = math.ceil((len(label_idx) - zero) * 1.000/ thread_num) 
	pids = {}

	#get_fea(0, len(label_idx), 5, fail_list)
	
	for i in range(thread_num):
	    pid = os.fork()
	    if pid == 0:
		pass
	    else:
	 	pids[pid] = i
		break

#	get_fea(0, len(label_idx), 0, fail_list)
	if pid != 0:
	    #child process
	    i = pids[pid]
	    start = zero + i * step
	    end = zero + (i + 1) * step
	    print 'PID:', pid, 'start:', start, 'end:', end
	    get_fea(start, end, i, fail_list)
