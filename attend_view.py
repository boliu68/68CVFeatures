#Author: bliuab, the HKUST
#Date: 2013.12.31
from cv2 import *
import numpy as np

def attend_view(saliency):

        CM = np.sum(saliency)

	x0 = np.sum([np.sum(saliency[i, :] * i) for i in range(saliency.shape[0])]) / CM
	y0 = np.sum([np.sum(saliency[:, j] * j) for j in range(saliency.shape[1])]) / CM

	w_ = np.sum([np.sum(saliency[i, :] * abs(i - x0)) for i in range(saliency.shape[0])]) / CM
	h_ = np.sum([np.sum(saliency[:, j] * abs(j - y0)) for j in range(saliency.shape[1])]) / CM

	alpha = 1

	W = 2 * alpha * w_
	H = 2 * alpha * h_

        return [int(x0), int(y0), int(W), int(H)]
