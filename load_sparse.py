import os
import sys
import string
import numpy as np
import scipy
import scipy.io

def load_sparse(file_name):
    rows = [] #行号
    cols = [] #列号
    data = [] #数据值
    row = 0 # 行标
    with open(file_name) as f:
        for line in f:
            word_index, url, title, para, base64, word_seg, top100 = line.strip('\n').split('\t')
            top100 = top100.split(' ')
            topk = 0
            for col in top100:
                topk += 1
                rows.append(row) #行标
                cols.append(int(col) - 1) #列标
                data.append((10 - topk)/55.0) #数据
                if topk >= 10: break #只选top10近邻表示
            row += 1
    tfidf = scipy.sparse.csr_matrix((data, (rows, cols)), (4409556, 4409556)) #稀疏矩阵的size m*m
    return tfidf


