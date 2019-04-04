# -*- coding: utf-8 -*-
"""

@author: yingwenjie

"""
import scipy
import numpy as np
from Normalize import normalize
from utils import *

def BCSH1(weight, bits=64, iters=50, lambd=0.1):
    X = weight.transpose()
    X[X > 0] = 1
    X1 = X.copy()
    X0 = 1 - X
    X1 = normalize(X1)
    X0 = normalize(X0)
    [n,m] = X.shape

    B = np.random.randint(0,2,(bits,m))
    B[B == 0] = -1

    #PB1 = np.zeros((bits,1))
    for i in range(iters):

        tempB1 = B.copy()
        tempB0 = -B
        tempB1[tempB1 < 0] = 0
        tempB0[tempB0 < 0] = 0

        PX1_B1 = (np.dot(tempB1,X.transpose())+1)/(m+2)
        PX1_B0 = (np.dot(tempB0,X.transpose())+1)/(m+2) 
        PB1 = np.sum(tempB1,1,keepdims=True)/m
        
        PX0_B1 = 1 - PX1_B1
        PX0_B0 = 1 - PX1_B0
        PB0 = 1-PB1

        
        logPX1_B1 = np.log2(PX1_B1)
        logPX1_B0 = np.log2(PX1_B0)
        logPX0_B1 = np.log2(PX0_B1)
        logPX0_B0 = np.log2(PX0_B0)

        logPB1 = np.dot(logPX1_B1,X1) + np.dot(logPX0_B1,X0) 
        logPB0 = np.dot(logPX1_B0,X1) + np.dot(logPX0_B0,X0)             

        tmp = logPB1 - logPB0
        tmp[tmp > 32] = 32
        PXB1 = np.power(2,tmp)
        PXB1 = PXB1 / (1 + PXB1)
        Fx = PXB1 * 2 -1

        Y = Update(B,bits)

        old_B = B.copy()
        for i in range(bits):
            for j in range(m):
                if((np.power((1 - Fx[i,j]),2) + lambd * np.power((1 - Y[i,j]),2)) <= (np.power((-1 - Fx[i,j]),2) + lambd * np.power((-1 - Y[i,j]),2))):
                    B[i,j] = 1
                else:
                    B[i,j] = -1
        updateB = sum(sum(B != old_B))
        print('update-------------------')
        print(updateB)
     
    scipy.io.savemat('./argfile/arg1.mat',{'B': B,'logPX1_B1': logPX1_B1,'logPX1_B0': logPX1_B0,'logPX0_B1': logPX0_B1,'logPX0_B0': logPX0_B0})
    
    return B
