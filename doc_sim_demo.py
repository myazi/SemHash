# -*- coding: utf-8 -*-
"""

@author: yingwenjie

"""

import os
import sys
import string
import numpy as np
import scipy.io

if __name__ == "__main__" :
	docs = [] 
	with open('./argfile/doc.utf8','r') as f:
		for line in f:
			docs.append(list(line.strip('\n').split(',')))	
	print(len(docs))	
	arg = scipy.io.loadmat('./argfile/arg2.mat')
	B = arg['B']
	Sim = np.dot(B.transpose(),B)
	TopK = int(20)
	MIN = -10000000
	for i in range(Sim.shape[0]):
		k = 0
		print('-----' + "".join(docs[i]) + '-----')
		while k < TopK:
			max_index = np.argmax(Sim[i,:])
			print("".join(docs[max_index]) + '  ' + str(Sim[i,max_index]) +  '\n')
			Sim[i,max_index] = MIN
			k = k + 1
