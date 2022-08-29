# -*- coding: utf-8 -*-
"""

@author: yingwenjie

"""
import scipy 
import numpy as np
from Normalize import *
from utils import *

def BCSH2(weight, bits=64, iters=100, lambd=0.1):
    X = weight.transpose()    
    X = normalize(X)
    [n, m] = X.shape
    print(n, m)

    B = np.random.randint(0, 2, (bits, m))
    B[B == 0] = -1

    for it in range(iters):
        tempB1 = B.copy()
        tempB0 = -B
        tempB1[tempB1 < 0] = 0
        tempB0[tempB0 < 0] = 0

        PX1_B1 = np.dot(tempB1, X.transpose())
        ALL1 = np.sum(PX1_B1, 1, keepdims=True)
        for r in range(bits):
            PX1_B1[r, :] = (PX1_B1[r, :] + 1) / (ALL1[r, 0] + n)
        
        PX1_B0 = np.dot(tempB0,X.transpose())
        ALL0 = np.sum(PX1_B0, 1, keepdims=True)
        for r in range(bits):
            PX1_B0[r, :] = (PX1_B0[r, :] + 1) / (ALL0[r, 0] + n)
        
        PB1 = np.sum(tempB1, 1, keepdims=True) / m

        logPX1_B1 = np.log2(PX1_B1)
        logPX1_B0 = np.log2(PX1_B0)

        logPB1 = np.dot(logPX1_B1, X)
        logPB0 = np.dot(logPX1_B0, X)

        tmp = (logPB1 - logPB0)  ### 规范化很重要，特征进行规范化
        tmp[tmp > 32] = 32
        
        PXB1 = np.power(2,tmp)
        PXB1 = PXB1 / (1 + PXB1)
        Fx = PXB1 * 2 -1
        
        Y = Update(B, bits)
        old_B = B.copy()
        for i in range(bits):
            for j in range(m):
                if((np.power((1 - Fx[i, j]), 2) + lambd * np.power((1 - Y[i, j]), 2)) <= (np.power((-1 - Fx[i, j]), 2) + lambd * np.power((-1 - Y[i, j]), 2))):
                    B[i, j] = 1
                else:
                    B[i, j] = -1
        updateB = sum(sum(B != old_B))
        print('update-----------------' + str(lambd))
        print(updateB)

        loss1 = np.trace(np.dot((B - Fx), (B - Fx).transpose()))
        loss2 = np.trace(np.dot((B - Y), (B - Y).transpose()))
        Loss = loss1 + lambd * loss2
        print('Loss=' + str(Loss))    
    scipy.io.savemat('./argfile/arg.mat',{'B': B,'logPX1_B1': logPX1_B1,'logPX1_B0': logPX1_B0})
    
    return B
