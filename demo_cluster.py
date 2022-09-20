# -*- coding: utf-8 -*-
"""

@author: yingwenjie

"""
import sys
import os
import numpy as np
import scipy.io
import numpy as np
import re
import math
from utils import *
from itertools import chain

line2index = {}
index2index = {}

def print_line_index(task_dir="./data/tmp"):
    of = open(task_dir + "/seg_file_orgin_hash", 'w')
    with open(task_dir + "/seg_file_orgin") as f:
        for line in f:
            index = line2index[i]
            index = index2index.get(index,index)
            of.write(str(index) + "\t" + line)
    of.close() 

def merge_index(hash_list, train_num, K, bits):
    """
    hash_list: 按簇大小排序后，哈希码下doc的list
    train_num: 训练数据量
    K: megre后最终输出的类别数
    bits: 哈希码长度
    """
    avg_num = train_num / K
    print("agv_num: " + str(avg_num))
    top = len(hash_list)
    while top >= K:
        top -= 1
        index_top = hash_list[top][0]
        min_dist = 100
        min_k = top
        min_k_index = 0
        for i in range(K):
            index_len = len(hash_list[i][1])
            if index_len > avg_num: continue 
            index_k = hash_list[i][0]
            dist = bin(index_top ^ index_k).count('1')
            if dist < min_dist:
                min_dist = dist
                min_k = i
                min_k_index = index_k
        if min_k != top:
            for i in hash_list[top][1]:
                hash_list[min_k][1].append(i + "\t" + index2hash(index_top, bits))
            hash_list[top][1].clear()
            index2index[index_top] = min_k_index
    return hash_list

if __name__ == "__main__" :
    task_name = sys.argv[1]
    task_dir = "./data/" + task_name
    docs = []
    with open(task_dir + "/seg_file_orgin") as f: ##原始数据 + 分词结果文件
        for line in f:
            docs.append(list(line.strip().split("\t")))
    print("docs num: " + str(len(docs)))

    arg = scipy.io.loadmat(task_dir + "/arg.mat")
    B_index = arg['B_index']
    bits = arg['logPX1_B1'].shape[0]
    print("B_index shape: " + str(B_index.shape))

    hashs = {}
    for i in range(len(docs)):
        index = B_index[0, i]
        line2index[i] = index
        hashs.setdefault(index,[])
        hashs[index].append("\t".join(docs[i]))
        #hashs[index].append(docs[i])
    print("cluster num: " + str(len(hashs)))

    hash_list = sorted(hashs.items(), key = lambda x:len(x[1]), reverse = True) #按簇大小排序 
    #hash_list = sorted(hashs.items(), key = lambda x:(x[0]), reverse = True) #按哈希码值排序
    
    #hash_list = merge_index(hash_list, len(docs), 100, bits) #对原始哈希聚簇进行merge
    #print_line_index(task_dir) ##按原始数据行号打印出数据的哈希码
    
    for item in hash_list:
        index = item[0]
        hash_code = index2hash(index, bits)
        if len(hashs[index]) == 0: continue
        print('----------------------------------------------' + str(index) + ':' + hash_code  + '@' + str(len(hashs[index])) + '----------------------------------------------' )
        hashs[index].sort()
        for docsi in hashs[index]:
            print(str(index) + '\t' + str(docsi).replace("\n", ""))
