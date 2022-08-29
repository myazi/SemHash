# -*- coding: utf-8 -*-
"""

@author: yingwenjie

"""
import scipy 
import scipy.io 
import numpy as np
from Normalize import *
from utils import *
from evaluation import *
from Supervise import *

def save_arg(B,weight,bits):
    [m,n] = weight.shape
    """
    得到哈希码后，计算所有位的参数
    """
    B_copy = B.copy()
    tempB1 = B_copy
    tempB0 = -B_copy
    tempB1[tempB1 < 0] = 0
    tempB0[tempB0 < 0] = 0

    PX1_B1 = scipy.sparse.csr_matrix(tempB1) * weight
    PX1_B1 = PX1_B1.toarray().astype(float)
    ALL1 = np.sum(PX1_B1,1,keepdims=True)
    PX1_B1 = (PX1_B1 + 1)/(ALL1 + n)
            
    PX1_B0 = scipy.sparse.csr_matrix(tempB0) * weight
    PX1_B0 = PX1_B0.toarray().astype(float)
    ALL0 = np.sum(PX1_B0,1,keepdims=True)
    PX1_B0 = (PX1_B0 + 1)/(ALL0 + n)

    logPX1_B1 = np.log2(PX1_B1)
    logPX1_B0 = np.log2(PX1_B0)
    print(logPX1_B1.shape)
    print(logPX1_B0.shape)
    
    B_copy[B_copy < 0] = 0
    Bs = B_copy
    B_index = []
    for i in range(m):
        index = 0
        for j in range(bits):
            index += 2** (bits - j - 1) * Bs[j,i]
        B_index.append(index)
    B_index = np.array(B_index)
    scipy.io.savemat('./data/argfile/arg2.mat',{'B_index':B_index,'logPX1_B1': logPX1_B1,'logPX1_B0': logPX1_B0})    

#def BCSH_paper(B, tfidf, label, bits=32, iters=10):
def BCSH_paper(B, tfidf, label, B_test, tfidf_test, label_test, B_dev, tfidf_dev, label_dev, bits=32, iters=100):
    weight = tfidf
    [m, n] = weight.shape
    print(m, n)
    weight_test = tfidf_test
    weight_dev = tfidf_dev
    #B = np.random.randint(0,2,(bits,m)) #随机初始化哈希码B
    B[B < 0] = -1
    B[B >= 0] = 1
    B_test[B_test < 0] = -1
    B_test[B_test >= 0] = 1
    B_dev[B_dev < 0] = -1
    B_dev[B_dev >= 0] = 1

    best_loss = float('inf')
    for it in range(iters):
        print("===============" + str(it) + "================")
        alpha3 = 0
        for s in range(1):
            print(B)
            B_grad,loss = train1(B,label,B_test,label_test,B_dev,label_dev)
            print(B_grad)
            B = B - alpha3 * B_grad.t().numpy()
            B[B < 0] = -1
            B[B >= 0] = 1
            print(B)
        #if loss < best_loss:
        #    best_loss = loss
        #    save_arg(B,weight,bits)
        for r in range(bits):
            B_bit = B[r,:].reshape(1,m)
            tempB0 = -B[r,:].reshape(1,m)
            tempB1 = -tempB0
            tempB0[tempB0 < 0] = 0
            tempB1[tempB1 < 0] = 0

            PX1_B1 = scipy.sparse.csr_matrix(tempB1) * weight
            PX1_B1 = PX1_B1.toarray().astype(float)
            ALL1 = np.sum(PX1_B1,1,keepdims=True)
            PX1_B1 = (PX1_B1 + 1)/(ALL1 + n)
            
            PX1_B0 = scipy.sparse.csr_matrix(tempB0) * weight
            PX1_B0 = PX1_B0.toarray().astype(float)
            ALL0 = np.sum(PX1_B0,1,keepdims=True)
            PX1_B0 = (PX1_B0 + 1)/(ALL0 + n)
            
            logPX1_B1 = np.log2(PX1_B1)
            logPX1_B0 = np.log2(PX1_B0)

            logPB1 = weight * scipy.sparse.csr_matrix(logPX1_B1.transpose())
            logPB0 = weight * scipy.sparse.csr_matrix(logPX1_B0.transpose())
            logPB1 = logPB1.toarray().transpose()
            logPB0 = logPB0.toarray().transpose()

            tmp = (logPB1 - logPB0)  ### 规范化很重要
            tmp[tmp > 32] = 32
            
            PXB1 = np.power(2,tmp)
            PXB1 = PXB1 / (1 + PXB1)
            Fx = PXB1 * 2 -1

            ##位平衡约束
            alpha1 = 0.5
            bit_balance = np.sum(B_bit,axis=1).reshape(1,1) / m
            print(bit_balance[0,0])
            Fx -= alpha1 * bit_balance 

            ##位独立约束
            alpha2 = 0.1
            bit_un = np.dot(B_bit, B.transpose()).reshape(1, bits) / m #* r #逐位学习位约束逐渐加强
            for i in range(r,bits):
                bit_un[0,i] = 0
            bit_un_res = np.dot(bit_un,B).reshape(1, m)
            Fx -= alpha2 * bit_un_res
            #print(bit_un_res)
            #Fx -= alpha3 * B_grad.t().numpy()[r,:]

            ## 监督梯度
            #train1(B,label)
            #old_B = B.copy() 
            #B[r,:] = Fx[0,:]
            B[r, Fx[0,:] < 0] = -1
            B[r, Fx[0,:] >= 0] = 1
            #updateB = sum(sum(B!=old_B))
            #print(updateB)

            #loss1 = np.trace(np.dot((B_bit - Fx),(B_bit - Fx).transpose()))
            #loss2 = np.sum(B_bit,axis=1).reshape(1,bits) / m
            #bit_un = np.dot(B, B.transpose()) / m
            #loss3 = np.trace(np.dot(bit_un,bit_un.transpose()))
            #Loss = loss1 + alpha1 * loss2[0,r] + alpha2 * loss3
            #print('Loss=' + str(Loss) + "\t" + "loss1: " + str(loss1) + "\t" + "loss2: " + str(loss2) + "\t" + "loss3: " + str(loss3))    
            #print('Loss=' + str(Loss) + "\t" + "loss1: " + str(loss1) + "\t" + "loss2: " + str(loss2) + "\t" + "loss3: " + str(loss3))    
        #eval_retrieval(B.transpose(), label.transpose(), top_n=100)

        loss2 = np.sum(B,axis=1) 
        loss3 = (np.dot(B,B.transpose()))
        loss4 = sum(sum(abs(np.dot(B,B.transpose())))) - np.trace(np.dot(B,B.transpose()))
        print(logPX1_B1.shape)
        print("loss2" + "\t" + str(loss2))
        print("loss3" + "\t" + str(loss3))
        print("loss4" + "\t" + str(loss4))

        """
 #      得到哈希码后，计算所有位的参数
        """
        B_copy = B.copy()
        tempB1 = B_copy
        tempB0 = -B_copy
        tempB1[tempB1 < 0] = 0
        tempB0[tempB0 < 0] = 0

        PX1_B1 = scipy.sparse.csr_matrix(tempB1) * weight
        PX1_B1 = PX1_B1.toarray().astype(float)
        ALL1 = np.sum(PX1_B1,1,keepdims=True)
        PX1_B1 = (PX1_B1 + 1)/(ALL1 + n)
    
        PX1_B0 = scipy.sparse.csr_matrix(tempB0) * weight
        PX1_B0 = PX1_B0.toarray().astype(float)
        ALL0 = np.sum(PX1_B0,1,keepdims=True)
        PX1_B0 = (PX1_B0 + 1)/(ALL0 + n)

        logPX1_B1 = np.log2(PX1_B1)
        logPX1_B0 = np.log2(PX1_B0)
        print(logPX1_B1.shape)
        print(logPX1_B0.shape)

        logPB1 = weight * scipy.sparse.csr_matrix(logPX1_B1.transpose())
        logPB0 = weight * scipy.sparse.csr_matrix(logPX1_B0.transpose())
        logPB1 = logPB1.toarray().transpose()
        logPB0 = logPB0.toarray().transpose()

        tmp = (logPB1 - logPB0)  ### 规范化很重要
        tmp[tmp > 32] = 32
            
        PXB1 = np.power(2,tmp)
        PXB1 = PXB1 / (1 + PXB1)
        B = PXB1 * 2 -1
        B[B < 0] = -1
        B[B >= 0] = 1

        logPB1 = weight_test * scipy.sparse.csr_matrix(logPX1_B1.transpose())
        logPB0 = weight_test * scipy.sparse.csr_matrix(logPX1_B0.transpose())
        logPB1 = logPB1.toarray().transpose()
        logPB0 = logPB0.toarray().transpose()

        tmp = (logPB1 - logPB0)  ### 规范化很重要
        tmp[tmp > 32] = 32
            
        PXB1 = np.power(2,tmp)
        PXB1 = PXB1 / (1 + PXB1)
        B_test = PXB1 * 2 -1
        B_test[B_test < 0] = -1
        B_test[B_test >= 0] = 1

        logPB1 = weight_dev * scipy.sparse.csr_matrix(logPX1_B1.transpose())
        logPB0 = weight_dev * scipy.sparse.csr_matrix(logPX1_B0.transpose())
        logPB1 = logPB1.toarray().transpose()
        logPB0 = logPB0.toarray().transpose()

        tmp = (logPB1 - logPB0)  ### 规范化很重要
        tmp[tmp > 32] = 32
            
        PXB1 = np.power(2,tmp)
        PXB1 = PXB1 / (1 + PXB1)
        B_dev = PXB1 * 2 -1
        B_dev[B_dev < 0] = -1
        B_dev[B_dev >= 0] = 1

        ##1->2
        logPX1_B1_B0 = (logPX1_B1 - logPX1_B0)
        logPX1_B1_B0_sign = np.zeros((logPX1_B1_B0.shape), dtype = int)
        logPX1_B1_B0_sign[logPX1_B1_B0 >= 0] = 1
        logPX1_B1_B0_sign[logPX1_B1_B0 < 0] = -1
        B_num = weight * logPX1_B1_B0_sign.transpose()
        B_test_num = weight_test * logPX1_B1_B0_sign.transpose()
        B_dev_num = weight_dev * logPX1_B1_B0_sign.transpose()
        print("1->2")
        print(B_dev_num)
        B_grad,loss = train1(B_num.T,label,B_test_num.T,label_test,B_dev_num.T,label_dev)

    B[B < 0] = 0
    Bs = B
    B_index = []
    for i in range(m):
        index = 0
        for j in range(bits):
            index += 2** (bits - j - 1) * Bs[j,i]
        B_index.append(index)
    B_index = np.array(B_index)
    scipy.io.savemat('./data/argfile/arg2.mat',{'B_index':B_index,'logPX1_B1': logPX1_B1,'logPX1_B0': logPX1_B0})    
    
    return Bs
