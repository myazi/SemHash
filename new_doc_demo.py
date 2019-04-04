#-*- coding: utf-8 -*-
"""

@author: yingwenjie

"""

import os
import sys
import string
import numpy as np
import scipy
from Normalize import *
import jieba
import jieba.posseg as pseg
from jieba import analyse
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import re
if __name__ == "__main__" : 

    arg2 = scipy.io.loadmat('./argfile/arg2.mat')
    B = arg2['B']
    logPX1_B1 = arg2['logPX1_B1']
    logPX1_B0 = arg2['logPX1_B0']
    
    docs = []
    with open('./argfile/doc.utf8','r') as f:
         for line in f:
             docs.append(line.strip())

    dics = []   
    with open('./argfile/word.utf8','r') as f:
         for line in f:
             dics.append(line.strip())	
  
    filename = sys.argv[1]
    f = open(filename,'+r',encoding='UTF-8') 
    doc = f.read()
    f.close()
    
    doc = re.sub('[A-Za-z0-9\!\%\[\]]',"",doc)
    print("Using jieba on " + filename)    
    
    #words = jieba.cut(doc,cut_all=True)
    
    tfidf = analyse.extract_tags
    words = tfidf(doc, topK=100)
    
    doc_vec = np.zeros((len(dics),1))
    for word in words:
        word = ''.join(word.split())
        if (word in dics):
           index = dics.index(word)
           doc_vec[index,0] =  1

    logPB1 = np.dot(logPX1_B1, doc_vec)
    logPB0 = np.dot(logPX1_B0, doc_vec)

    tmp = (logPB1 - logPB0)
    tmp[tmp >32] = 32
    
    PXB1 = np.power(2,tmp)
    PXB1 = PXB1 / (1 + PXB1)
    test_B = PXB1 *2 - 1
    
    test_B[test_B > 0] = 1
    test_B[test_B < 0] = -1
   
    sim = np.dot(B.transpose(),test_B)
    B[B < 0] = 0    
    test_B[test_B < 0 ] =0
     
    test_B_codes = ""
    for code in test_B[:,0]:
        test_B_codes = test_B_codes + str(int(code))
    print('---------------------------------------' + filename + '----------------------------------------')
    print(filename + ' codes is :' + test_B_codes)
 
    MIN = -10000000
    topK = 20
    k = 0
    while k < topK:
        max_index = np.argmax(sim[:,0])
        B_codes = ""
        for code in B[:,max_index]:
            B_codes = B_codes + str(int(code))
        print(B_codes + '    ' + docs[max_index] + ':   ' + str(sim[max_index,0]))
        sim[max_index,0] = MIN
        k = k + 1
