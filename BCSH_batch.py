# -*- coding: utf-8 -*-
"""

@author: yingwenjie

"""
import scipy 
import scipy.io 
import numpy as np
from utils import *

def BCSH_batch(tfidf, bits=2, rand_time=8, batch=100000, iters=100, lambd=0.0005, word=[], arg_dir="./data/tmp"):

    [m,n] = tfidf.shape
    print(m,n)
    """
    外层循环哈希码，每次计算bits位的哈希码【每次计算的bits哈希码是独立的，且内层迭代完后不再更新】
    内层循环为计算bits位哈希码进行的收敛次数
    """
    summy = np.ones((1,n))
    for i in range(rand_time):
        print("===============" + str(i) + "================")

        """
        shuffler
        """
        col_rand_array = np.arange(tfidf.shape[0])
        np.random.shuffle(col_rand_array)
        weight = tfidf[col_rand_array[0:batch],:]
        [m,n] = weight.shape
        print(m,n)

        """
        dropout 
        1、概率dropout高频词 
        2、随机dropout高频词
        """
        for k in range(len(word)):
            print(str(k) + "\t" + str(summy[0,k]) + "\t" + str(word[k])) #打印高频词及其概率
        num = np.random.randint(0,2,(m,1))
       
        #weight = weight / summy #概率dropout
        #weight = scipy.sparse.csr_matrix(weight)
        #summy[summy >= 3] = 0
        #summy[summy < 3] = 1
        #dropout = scipy.sparse.csr_matrix(num) * scipy.sparse.csr_matrix(summy)  
        #print(weight)
        #weight = weight.multiply(dropout)
        #print(weight)
        #print(weight.nonzero())
        #weight = weight.tolil()
        #for k in range(n):
        #    if summy[0,k] > 10:
        #       weight[:,k] *= 0.1

        """
        随机初始化哈希码B
        """
        B = np.random.randint(0,2,(bits,m))
        B[B == 0] = -1

        for it in range(iters):
            tempB1 = B.copy()
            tempB0 = -B
            tempB1[tempB1 < 0] = 0
            tempB0[tempB0 < 0] = 0

            PX1_B1 = scipy.sparse.csr_matrix(tempB1) * weight
            PX1_B1 = PX1_B1.toarray().astype(float)
            ALL1 = np.sum(PX1_B1,1,keepdims=True)
            for r in range(bits):
                PX1_B1[r,:] = (PX1_B1[r,:]+1)/(ALL1[r,0]+n)
            
            PX1_B0 = scipy.sparse.csr_matrix(tempB0) * weight
            PX1_B0 = PX1_B0.toarray().astype(float)
            ALL0 = np.sum(PX1_B0,1,keepdims=True)
            for r in range(bits):
                PX1_B0[r,:] = (PX1_B0[r,:]+1)/(ALL0[r,0]+n)

            """
            归一化高频词在0，1哈希码下的条件概率, 效果不行
            """
            #PX1 = PX1_B1 + PX1_B0
            #PX1_B1 = PX1_B1 / PX1 
            #PX1_B0 = PX1_B0 / PX1 
            #PB1 = np.sum(tempB1,1,keepdims=True)/m #计算哈希码为1的概率，查看位平衡效果


            logPX1_B1 = np.log2(PX1_B1)
            logPX1_B0 = np.log2(PX1_B0)

            logPB1 = weight * scipy.sparse.csr_matrix(logPX1_B1.transpose())
            logPB0 = weight * scipy.sparse.csr_matrix(logPX1_B0.transpose())
            logPB1 = logPB1.toarray().transpose()
            logPB0 = logPB0.toarray().transpose()

            tmp = (logPB1 - logPB0)  ### 规范化非常重要
            tmp[tmp > 32] = 32
            
            PXB1 = np.power(2,tmp)
            PXB1 = PXB1 / (1 + PXB1)
            Fx = PXB1 * 2 -1
            
            #Y = Update(B,bits)
            old_B = B.copy()

            ##位平衡约束
            alpha1 = 0.0
            bit_sum = np.sum(B,axis=1)
            bit_sum = bit_sum.reshape(bits,1)
            m_one = np.full((1,m),1.0/m)
            bit_balance = np.dot(bit_sum,m_one)
            Fx -= alpha1 * bit_balance 
            #Fx_ban = -bit_balance 

            ##位独立约束
            alpha2 = 0.0
            if i != 0:
               bit_un = np.dot(Bs, B.transpose())
               bit_un = bit_un.reshape(i * bits, bits)
               bit_un_res = np.dot(bit_un.transpose(),Bs)
               bit_un_res = bit_un_res.reshape(bits,m)
               Fx -= alpha2 * bit_un_res / m
               #Fx_un = bit_un_res / (m)
               #bit_un_sum = np.sum(bit_un, axis=0)
               #bit_un_sum = bit_un_sum.reshape(bits,1)
               #m_one = np.full((1,m),1.0/m)
               #bit_un = np.dot(bit_un_sum,m_one)
               #Fx -= alpha2 * bit_un 
            #if i == 0:
            #   Fx_un = np.zeros((bits, m))
            #for ii in range(bits):
            #    for j in range(m):
                    #if((np.power((1 - Fx[ii,j]),2) + alpha1 * np.power((1 - Fx_ban[ii,j]),2) + alpha2 * np.power((1 - Fx_un[ii,j]), 2) ) <= (np.power((-1 - Fx[ii,j]),2) + alpha1 * np.power((-1 - Fx_ban[ii,j]),2)) + alpha2 * np.power((-1 - Fx_un[ii,j]), 2)):
            #        if(np.power((1 - Fx[ii,j]),2) + alpha2 * np.power((1 - Fx_un[ii,j]), 2)) <= (np.power((-1 - Fx[ii,j]),2) + alpha2 * np.power((-1 - Fx_un[ii,j]), 2)):
            #            B[ii,j]=1
            #        else:
            #            B[ii,j]=-1

            B[Fx < 0] = -1
            B[Fx > 0] = 1
            updateB = sum(sum(B!=old_B))

            print('update-----------------' + str(lambd))
            print(updateB)

            loss1 = np.trace(np.dot((B - Fx),(B - Fx).transpose()))
            #loss2 = np.trace(np.dot((B - Y),(B - Y).transpose()))
            #loss2 = np.trace(np.dot((B - Fx_ban),(B - Fx_ban).transpose()))
            #loss3 = np.trace(np.dot((B - Fx_un),(B - Fx_un).transpose()))
            loss2 = np.sum(B,axis=1)
            loss3 = 0
            if i !=0:
               bit_un = np.dot(Bs, B.transpose())
               loss3 = np.trace(np.dot(bit_un,bit_un.transpose()))
            Loss = loss1 + alpha1 * loss2 + alpha2 * loss3
            print('Loss=' + str(Loss) + "\t" + "loss1: " + str(loss1) + "\t" + "loss2: " + str(loss2) + "\t" + "loss3: " + str(loss3))

        if(i==0):
            logPX1_B1s = logPX1_B1
            logPX1_B0s = logPX1_B0
            Bs = B
        else:
            logPX1_B1s = np.r_[logPX1_B1s,logPX1_B1] 
            logPX1_B0s = np.r_[logPX1_B0s,logPX1_B0]
            Bs = np.r_[Bs,B]
        PX1_B0s = np.power(2,logPX1_B0s)
        PX1_B1s = np.power(2,logPX1_B1s)
        #summy = np.sum(PX1_B1s + PX1_B0s,0).reshape(1,n) * n / (i + 1)
    loss2 = np.sum(Bs, axis=1)
    loss3 = (np.dot(Bs, Bs.transpose()))
    loss4 = sum(sum(abs(np.dot(Bs,Bs.transpose())))) - np.trace(np.dot(Bs, Bs.transpose()))
    print("loss2" + "\t" + str(loss2))
    print("loss3" + "\t" + str(loss3))
    print("loss4" + "\t" + str(loss4))

    oldi = 0
    mini_batch = int(batch / 20)
    for i in range(1,tfidf.shape[0] + 1):
        if i % mini_batch == 0:
           weight = tfidf[oldi:i,:]
           logPB1 = weight * scipy.sparse.csr_matrix(logPX1_B1s.transpose())
           logPB0 = weight * scipy.sparse.csr_matrix(logPX1_B0s.transpose())
           logPB1 = logPB1.toarray().transpose()
           logPB0 = logPB0.toarray().transpose()

           tmp = (logPB1 - logPB0)  ### 规范化很重要
           tmp[tmp > 32] = 32
    
           PXB1 = np.power(2,tmp)
           PXB1 = PXB1 / (1 + PXB1)
           Fx = PXB1 * 2 - 1

           B = np.zeros(shape=Fx.shape)
           B[Fx < 0] = -1
           B[Fx > 0] = 1
           oldi = i
           if (i == mini_batch):
              Bs = B
           else:
              Bs = np.c_[Bs,B]
        i += 1
    if i % mini_batch != 0:
       weight = tfidf[oldi:i,:]
       logPB1 = weight * scipy.sparse.csr_matrix(logPX1_B1s.transpose())
       logPB0 = weight * scipy.sparse.csr_matrix(logPX1_B0s.transpose())
       logPB1 = logPB1.toarray().transpose()
       logPB0 = logPB0.toarray().transpose()
       
       tmp = (logPB1 - logPB0)  ### 规范化很重要
       tmp[tmp > 32] = 32

       PXB1 = np.power(2,tmp)
       PXB1 = PXB1 / (1 + PXB1)
       Fx = PXB1 * 2 - 1

       B = np.zeros(shape=Fx.shape)
       B[Fx < 0] = -1
       B[Fx > 0] = 1
       if (i < mini_batch):
         Bs = B
       else:
         Bs = np.c_[Bs,B]
    scipy.io.savemat(arg_dir + '/arg.mat',{'B': Bs,'logPX1_B1': logPX1_B1s,'logPX1_B0': logPX1_B0s})    
    return Bs
