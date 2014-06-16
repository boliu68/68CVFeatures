#Author: bliuab, the HKUST
#Date: 2014.1.2
from cv2 import *
import numpy as np
import math

def get_colemotion(img):

    HSL = cvtColor(img, COLOR_BGR2HSV)

    L = HSL[:, :, 2]
    H = np.array(HSL[:, :, 0], np.int16) * 2
    S = np.array(HSL[:, :, 1], np.int16)

    c = [0.0] * 7

    c[0] = np.min(L)
    c[6] = np.max(L)
    for j in range(1, 6):
        c[j] = int( c[0]+ j * (c[6] * 1.0000 - c[0] * 1.0000) / 6)
    
   
    iteration_num = 0
    while True:
        U = np.zeros((img.shape[0], img.shape[1], 5))

        rule1_idx = (L <= c[1])
        U[rule1_idx, :] = 0
        U[rule1_idx, 0] = 1

        rule2_idx = (L > c[5])
        U[rule2_idx, :] = 0
        U[rule2_idx, 4] = 1

        rule3_idx = np.zeros((6, U.shape[0], U.shape[1]), np.bool)

        rule3_idx[1] = (L > c[1]) * (L <= c[2])
        rule3_idx[2] = (L > c[2]) * (L <= c[3])
        rule3_idx[3] = (L > c[3]) * (L <= c[4])
        rule3_idx[4] = (L > c[4]) * (L <= c[5])

        #U[rule3_idx1, :] = 1 - U[rule3_idx1]
        for j in range(1, 5):
            U[rule3_idx[j], j - 1] = (c[j+1] - L[rule3_idx[j]]) * 1.0 / (c[j + 1] - c[j])
            U[rule3_idx[j], j] = 1 - U[rule3_idx[j], j - 1]
            for k in range(5):
                if k == j - 1 or k == j:
                    continue
                U[rule3_idx[j], k] = 0#1 - U[rule3_idx[j], j - 1]

        old_c = c[:]

        for j in range(1, 6):
	    U_sum = np.sum(U[:, :, j - 1]);
	    if U_sum == 0:
		U_sum = 1
            c[j] = np.sum(U[:, :, j - 1] * L) / U_sum 
            if np.isnan(c[j]):
                return [False]
	    c[j] = int(c[j])

        if c == old_c or iteration_num > 50:
	    #print 'Num of iteration:', iteration_num
	    break
	iteration_num += 1

    #2 fuzzy member function
    Uwarm = np.zeros(H.shape)
    Ucold = np.zeros(H.shape)

    warm_idx = (H <= 140) + ((H >= 320) * (H <= 360))
    cold_idx = (H >= 140) * ( H <= 320)

    Uwarm[warm_idx] = np.cos((H[warm_idx] - 50) * math.pi / 180)
    Ucold[cold_idx] = np.cos((H[cold_idx] - 230)* math.pi / 180)

    Ures = np.zeros((Uwarm.shape[0], Uwarm.shape[1], 10), np.float32)
    histogram1 = [0] * 10

    for i in range(5):
        Ures[:, :, 2 * i] = Uwarm * U[:, :, i]
        Ures[:, :, 2 * i + 1] = Ucold * U[:, :, i]

    max_prob = np.argmax(Ures, 2)
    #print max_prob.shape

    for i in range(10):
        histogram1[i] = np.sum(max_prob == i)

    #print histogram

    Lab_con = math.sqrt(np.sum((H - np.mean(H)) ** 2 + (S - np.mean(S)) ** 2) / (img.shape[0] * img.shape[1]))

    S = S * 100 / 255
    UC = np.zeros((img.shape[0], img.shape[1], 3))

    UC[S < 10, 0] = 1
    UC[(S >= 10) * (S <= 27), 0] = (27 - S[(S >= 10) * (S <= 27)]) / 17
    UC[S > 27, 0] = 0

    UC[(S >= 10) * (S < 27), 1] = (S[(S >= 10) * (S < 27)] - 10) / 17
    UC[(S >= 27) * (S < 51), 1] = (51 - S[(S >= 27) * (S < 51)]) / 24
    UC[S >= 51, 1] = 0

    UC[S < 27, 2] = 0
    UC[(S >= 27) * (S <= 51), 2] = (S[(S >= 27) * (S <= 51)] - 27) / 24
    UC[S > 51, 2] = 1

    histogram2 = [0] * 7
    Ures2 = np.zeros((img.shape[0], img.shape[1], 6))

    for i in range(3):
        Ures2[:, :, 2 * i] = Uwarm * UC[:, :, i]
        Ures2[:, :, 2 * i + 1] = Ucold * UC[:, :, i]

    max_prob2 = np.argmax(Ures2, 2)
    for i in range(6):
        histogram2[i] = np.sum(max_prob2 == i)

    histogram2[6] = Lab_con

    hist = histogram1 + histogram2

    return hist

