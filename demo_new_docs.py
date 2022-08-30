#-*- coding: utf-8 -*-
"""

@author: yingwenjie

"""

import os
import sys
import string
import numpy as np
import scipy
import scipy.io
import jieba
import jieba.posseg as pseg
from jieba import analyse
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import re
from utils import *

if __name__ == "__main__" : 

    task_name = sys.argv[1]
    filename = sys.argv[2]

    arg = scipy.io.loadmat("./data/" + task_name + "/arg.mat")
    B_index = arg['B_index']
    B_index = B_index[0,:].tolist()
    logPX1_B1 = arg['logPX1_B1']
    logPX1_B0 = arg['logPX1_B0']
    bits = logPX1_B1.shape[0] #哈希码长度
    
    train_docs = []
    with open("./data/" + task_name + "/seg_file_orgin") as f:
         for line in f:
             train_docs.append(line.strip())

    dics = []   
    with open("./data/" + task_name + "/arg_word.utf8",'r') as f:
         for line in f:
             dics.append(line.strip())	
  
    test_docs = []
    test_seg_list = []
    with open(filename) as f:
        for line in f: 
            sample  = line.strip('\n')
            #sample  = re.sub('[A-Za-z0-9\!\%\[\]]',"",line.strip('\n'))
            #print("Using jieba on " + filename)    
            #words = jieba.cut(doc,cut_all=True)
            tfidf = analyse.extract_tags
            seg_list = tfidf(sample, topK=5)
            test_docs.append(sample)
            test_seg_list.append(seg_list)
    print(test_docs)
    doc_vec = np.zeros((len(dics),len(test_seg_list)))
    for i in range(len(test_seg_list)):
        word = test_seg_list[i]
        for w in word:
            if w in dics:
               index = dics.index(w)
               doc_vec[index,i] =  1

    logPB1 = np.dot(logPX1_B1, doc_vec)
    logPB0 = np.dot(logPX1_B0, doc_vec)

    tmp = (logPB1 - logPB0)
    tmp[tmp >32] = 32
    
    PXB1 = np.power(2,tmp)
    PXB1 = PXB1 / (1 + PXB1)
    test_B = PXB1 *2 - 1
    
    test_B[test_B > 0] = 1
    test_B[test_B < 0] = 0

    B_index_dict = {}
    for i in range(len(B_index)):
        B_index_dict.setdefault(B_index[i],[])
        B_index_dict[B_index[i]].append(train_docs[i])

    for i in range(test_B.shape[1]):
        index = 0
        for j in range(bits):
            index += 2**(bits - j - 1) * test_B[j,i]
        index = int(index)
        topK = 10
        for b in B_index_dict:
            dis = bin(index ^ b).count('1')
            if dis < 2:
                for docs in B_index_dict[b]:
                    print(str(index) + "\t" + "".join(test_docs[i]) + "\t" + str(b) + "\t" + "".join(docs) + "\t" + str(dis))
                    topK -= 1
                    if topK < 0: break
            if topK < 0: break
