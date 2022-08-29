# -*- coding: utf-8 -*-
"""

@author: yingwenjie

"""
import os
import sys
import string
import numpy as np
import scipy.io
from scipy.sparse import csr_matrix
from Keyword import *

def get_word_hash(wordfile, argfile):
    """
    获取预训练参数中词的哈希码
    """
    words = [] 
    word2index = {}
    word2hash = {}
    with open(wordfile,'r') as f:
        i = 0
        for line in f:
            word = line.strip('\n')
            words.append(word)    
            word2index[word] = i
            i += 1
    arg = scipy.io.loadmat(argfile)
    bits = logPX1_B1.shape[0]
    logPX1_B1 = arg['logPX1_B1']
    logPX1_B1 = np.power(2,logPX1_B1)
    logPX1_B0 = arg['logPX1_B0']
    logPX1_B0 = np.power(2,logPX1_B0)
    logPX1_B1_B0 = (logPX1_B1 - logPX1_B0) / (logPX1_B1 + logPX1_B0) 
    logPX1_B1_B0_sign = np.zeros((logPX1_B1_B0.shape), dtype = int)
    logPX1_B1_B0_sign[logPX1_B1_B0 >= 0] = 1
    logPX1_B1_B0_sign[logPX1_B1_B0 < 0] = -1
    for i in range(len(words)):
        word2hash[words[i]] = logPX1_B1_B0_sign[:,i]
    return word2hash, word2index

def word2sample_hash(seg_file, word2hash, word2index):
    samples = []
    #print(type(word2hash["中国"]))
    #print((word2hash["中国"].shape))
    nnn = 0
    non = np.zeros((word2hash["中国"].shape[0]), dtype=np.float) ## zeros
    with open(seg_file) as f:
       for line in f:
           seg_list = line.strip('\n').split('\t')
           label = seg_list[0]
           del seg_list[0]
           code = np.zeros((word2hash["中国"].shape[0]), dtype=np.float)
           one_hot = []
           for word in seg_list:
               code += word2hash.get(word, non)
               index = word2index.get(word, -1) 
               if index != -1:
                   one_hot.append(index)
           if len(one_hot) > 0:
               samples.append(str(label) + "\t" + "_".join([str(i) for i in code]) + "\t" + "_".join([str(i) for i in one_hot]))
           else:
               nnn+=1
    print("nnn" + "\t" + str(nnn))
    return samples

def init_B(file_name, data_type):
    topK = 10
    max_features = 100000
    label_flag = 0
    if data_type == "thucnews" or data_type == "inews":
         topK = 50
    if data_type == "tnews":
       label_flag = 1

    seg_file = fenci(file_name,topK)
    #word2hash,word2index = get_word_hash('./data/argfile_wiki_128/word.utf8','./data/argfile_wiki_128/arg2.mat')
    word2hash,word2index = get_word_hash('./data/argfile_tmp/word.utf8','./data/argfile_tmp/arg2.mat')
    samples = word2sample_hash(seg_file, word2hash, word2index)
    labels = []
    Bs = []
    row = []
    col = []
    data = []
    i = 0
    ff = open("./data/segfile/X",'w')
    for ss in samples:
        label_x = ss.strip('\n').split("\t")
        label = int(label_x[0])
        x = label_x[1]
        one_hot = label_x[2]
        if label_flag == 1:
            if label <= 104:
                 label = label - 100
            if label > 105 and label < 111:
                 label = label - 101
            if label > 111:
                 label = label - 102
        #print(ss)
        labels.append(label)
        Bs.append([float(i) for i in x.strip().split("_")])
        icols = one_hot.strip().split("_")
        r_str = str(label) + "\t" + str(x)
        ff.write(r_str + "\n")
        for icol in icols:
            col.append(int(icol))
            row.append(int(i))
            data.append(int(1))
        i += 1
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    X = csr_matrix((data, (row, col)), shape=(len(samples), len(word2index)))
    return np.array(labels), np.array(Bs).T, X 
