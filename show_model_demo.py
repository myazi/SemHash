# -*- coding: utf-8 -*-
"""

@author: yingwenjie

"""
import os
import sys
import string
import numpy as np
import scipy.io
from Normalize import *
if __name__ == "__main__" :
	words = [] 
	with open('./argfile/word.utf8','r') as f:
		for line in f:
			words.append(list(line.strip('\n').split(',')))	
	print(len(words))	
	arg = scipy.io.loadmat('./argfile/arg2.mat')
	logPX1_B1 = arg['logPX1_B1']
	logPX1_B0 = arg['logPX1_B0']
	TopK = int(10)
	MIN = -10000000
	for i in range(logPX1_B1.shape[0]):
		k = 0
		print('-------------------第' + str(i) + '位哈希码下词分布--------------------')
		print('          1        ' + '              ' + '           0           ')
		while k < TopK:
			max_index1 = np.argmax(logPX1_B1[i,:])
			max_index0 = np.argmax(logPX1_B0[i,:])
			print(str(words[max_index1]) + str(logPX1_B1[i,max_index1]) +  '  ' + str(words[max_index0]) + ' ' + str(logPX1_B0[i,max_index0]))
			logPX1_B1[i,max_index1] = MIN
			logPX1_B0[i,max_index0] = MIN
			k = k + 1	
"""
	logPX1_B1 = normalize(logPX1_B1)
	logPX1_B0 = normalize(logPX1_B0)
	word_sim1 = np.dot(logPX1_B1.transpose(),logPX1_B1)
	word_sim0 = np.dot(logPX1_B0.transpose(),logPX1_B0)

	for i in range(word_sim1.shape[0]):
		word_sim1[i,i] = MIN
		word_sim0[i,i] = MIN

	TopK = int(1000)
	k = 0
	while k < TopK:
		max_index1 = np.argmax(word_sim1)
		max_index1_x = int(max_index1 / word_sim1.shape[1])
		max_index1_y = int(max_index1 % word_sim1.shape[0])
		
		max_index0 = np.argmax(word_sim0)
		max_index0_x = int(max_index0 / word_sim0.shape[1])
		max_index0_y = int(max_index0 % word_sim0.shape[0])
		print('------------words' + str(k)  + '---------------')
		print(str(words[max_index1_x]) + '  ' + str(words[max_index1_y]) + '  ' + str(word_sim1[max_index1_x,max_index1_y]))
		word_sim1[max_index1_x,max_index1_y] = MIN
		word_sim1[max_index1_y,max_index1_x] = MIN

		print(str(words[max_index0_x]) + '  ' + str(words[max_index0_y]) + '  ' + str(word_sim0[max_index0_x,max_index0_y]))
		word_sim0[max_index0_x,max_index0_y] = MIN
		word_sim0[max_index0_y,max_index0_x] = MIN

		k = k + 1
"""
