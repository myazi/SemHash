import os
import sys
import string
import numpy as np
import scipy
import scipy.io

def normalize(X):
    [n,m] = X.shape
    ##特征维度归一化
    #L2norm = np.sum(np.multiply(X,X),1,keepdims=True)
    #L2norm[L2norm == 0] =1
    #L2norm = np.sqrt(L2norm)
    #for i in range(n):
    #    X[i,:] = X[i,:] / L2norm[i,0]
    
    ##样本级别归一化
    L2norm = np.sum(np.multiply(X,X),0,keepdims=True)
    L2norm[L2norm == 0] =1
    L2norm = np.sqrt(L2norm)
    for i in range(m):
        X[:,i] = X[:,i] / L2norm[0,i]
    
    return X

def Update(Z, r):
    Z_bar = Z-Z.mean(axis=1)[:,np.newaxis]
    SVD_Z = np.linalg.svd(Z_bar, full_matrices=False)
    Q = SVD_Z[2].T
    if Q.shape[1] < r:
        Q = gram_schmidt(np.c_[Q, np.ones((Q.shape[0], r - Q.shape[1]))])
    P = np.linalg.svd(np.dot(Z_bar, Z_bar.T))[0]
    Z_new = np.sqrt(Z.shape[1]) * np.dot(P, Q.T)
    return Z_new

def gram_schmidt(X):
    Q, R = np.linalg.qr(X)
    return Q

def K(x, y):
    return x if np.abs(x) > 10e-8 else y

def cal_bin_dis(a, b):
    #return bin(a ^ b).count('1')
    return (a ^ b)

def index2hash(index, bits):
    hash_code = ""
    bits = bits - 1
    while bits >= 0:
        bit = int(index) // int(2**bits)
        index -= 2**bits * bit
        hash_code = hash_code + str(bit)
        bits -= 1
    return hash_code

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
