# -*- coding: utf-8 -*-
"""

@author: yingwenjie

"""
import os
import sys
import string
import numpy as np
import scipy.io
from utils import *

if __name__ == "__main__" :
    task_name = sys.argv[1]
    task_dir = "./data/" + task_name
    words = [] 
    with open(task_dir + "/arg_word.utf8",'r') as f:
        for line in f:
            words.append(line.strip('\n'))    
    print(len(words))

    arg = scipy.io.loadmat(task_dir + "/arg.mat")
    logPX1_B1 = arg['logPX1_B1']
    bits = logPX1_B1.shape[0]
    logPX1_B1 = np.power(2, logPX1_B1)
    logPX1_B0 = arg['logPX1_B0']
    logPX1_B0 = np.power(2, logPX1_B0)
    logPX1_B1_B0 = (logPX1_B1 - logPX1_B0)
    logPX1_B1_B0_sign = np.zeros((logPX1_B1_B0.shape), dtype = int)
    logPX1_B1_B0_sign[logPX1_B1_B0 >= 0] = 1
    logPX1_B1_B0_sign[logPX1_B1_B0 < 0] = -1

    """
    输出word2hash
    """
    for i in range(len(words)):
        word = words[i]
        word_hash = logPX1_B1_B0_sign[:, i]
        word_hash_str = "".join([str(i if i > 0 else 0) for i in word_hash])
        print(str(word) + "\t" + word_hash_str)

    """
    根据词的哈希码，打印词与词之间的相似词
    """
    Sim = np.dot(logPX1_B1_B0_sign.transpose(), logPX1_B1_B0_sign)
    TopK = int(100)
    MIN = -bits
    for i in range(Sim.shape[0]):
        Sim[i, i] = MIN ##去重对角线元素
    for i in range(Sim.shape[0]):
        k = 0
        res_list = []
        while k < TopK:
            max_index = np.argmax(Sim[i, :])
            if(Sim[i, max_index] < bits - 6): # 汉明距离在6以内
                break
            res_list.append("".join(words[max_index]) + ":" + str(Sim[i, max_index]))
            Sim[i, max_index] = MIN
            k = k + 1
        res_str = str(words[i]) + "\t" + ",".join([str(a) for a in res_list])
        print(res_str)

    """
    输出每一位哈希码二值下的高频词
    """
    TopK = int(100)
    MIN = 0
    for i in range(logPX1_B1.shape[0]):
        k = 0
        print('-------------------第' + str(i) + '位哈希码下词分布--------------------') # 第i位哈希码二值下top100高频词
        print('          1        ' + '\t' + '           0           ')
        while k < TopK:
            max_index1 = np.argmax(logPX1_B1_B0[i, :])
            max_index0 = np.argmin(logPX1_B1_B0[i, :])
            print(str(words[max_index1]) + ':' + str(logPX1_B1_B0[i, max_index1]) + '\t' + str(words[max_index0]) + ':' + str(logPX1_B1_B0[i, max_index0]))
            
            logPX1_B1_B0[i, max_index1] = MIN
            logPX1_B1_B0[i, max_index0] = MIN
            k = k + 1    
