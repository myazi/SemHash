# -*- coding: utf-8 -*-
"""

@author: yingwenjie

"""

import os
import sys
import time
import string
import numpy as np
import scipy.io
import random
from utils.utils import *

def demo_doc_sim(task_name):
    task_dir = "./data/" + task_name
    docs = [] 
    docs_local = [] 
    col_rand_list = []
    index = 0
    with open(task_dir +  "/seg_file_orgin") as f:
        for line in f:
            docs.append((line.strip('\n'))) ##全集数据    
            if int(random.random() * 10000) % 10000 == 9: ##待查询数据 
                col_rand_list.append(index)
                docs_local.append((line.strip('\n')))    
            index += 1
    print("all doc num: " + str(len(docs)))
    print("test doc num:" + str(len(docs_local)))
    col_rand_array = np.array(col_rand_list)
    col_rand_set = set(col_rand_list)

    arg = scipy.io.loadmat(task_dir + "/arg.mat")
    bits = arg['logPX1_B1'].shape[0]
    B_index = arg['B_index']
    B_index_local = B_index[:, col_rand_array]
    B_index = B_index[0, :].tolist()
    B_index_local = B_index_local[0, :].tolist()

    B_index_dict = {}
    B_index_local_dict = {}
    for i in range(len(B_index)): #将哈希码相同的文档放到一起，将查询相似文档转换为查询相似的哈希码
        if i in col_rand_set:
            B_index_local_dict.setdefault(B_index[i], [])
            B_index_local_dict[B_index[i]].append(docs[i])
        else:
            B_index_dict.setdefault(B_index[i], [])
            B_index_dict[B_index[i]].append(docs[i])
    print("cluster num: " + str(len(B_index_dict))) #打印哈希码不同取值数

    for k in range(len(B_index_local)):
        a = B_index_local[k]
        min_dis = bits
        minb = 0
        for b in B_index_dict:
            a = int(a)
            b = int(b)
            dis = bin(a  ^  b).count('1')
            if dis < min_dis and dis < 4: #返回最相似的哈希码，且必须最相似哈希码的距离在4内
                min_dis = dis
                minb = b
        if minb != 0:
            for i in range(min(10, len(B_index_dict[minb]))):
                print("".join(docs_local[k]) + "\t" +  "".join(B_index_dict[minb][i]) + '\t' + str(min_dis) + "\t" + str(a) + "\t" + str(minb))
