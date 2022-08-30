# -*- coding: utf-8 -*-
"""

@author: yingwenjie

"""
import os
import sys
import string
import numpy as np
import scipy
import scipy.io
from Keyword import *
from utils import *
from evaluation import *

from BCSH import *
from BCSH7 import *
from ITSH import *
from BCSH_paper import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from init_B import *
    
def SemHash_BCTH():
    (allfile,path) = getFilelist(sys.argv)
    #max_feat = int(sys.argv[2])
    #bits = sys.argv[3]
    #iters = sys.argv[4]
    #lambd = sys.argv[5]
    print(path)    
    print(allfile)
    
    f = open('./argfile/doc.utf8','+w')   
    for ff in allfile :
        f.write(ff + '\n')
    f.close()
    
    path = fenci(allfile,path)
    (word,weight) = Tfidf(path,allfile)

    f = open('./argfile/word.utf8', 'w+')
    for j in range(len(word)):
        f.write(word[j] + "\n")
    f.close()

    #scipy.io.savemat('X.mat',{'X': weight})  
    
    B = BCSH2(weight)
    print(B.shape)    
    Sim = np.dot(B.transpose(),B)
    B[B < 0] = 0
    f = open('./argfile/hashcode.utf8', '+w')
    for i in range(B.shape[1]):
        for j in range(B.shape[0]):
            f.write(str(B[j,i]))
        f.write('    ' + allfile[i])
        f.write('\n')
    for i in range(Sim.shape[0]):
        for j in range(Sim.shape[0]-i):
            f.write(allfile[i]+'      '+allfile[j]+'     '+str(Sim[i,j])+'\n')
    
    f.close()

def SemHash_BCTH7():
    file_name = sys.argv[1] # 训练集文件
    topK = int(sys.argv[2]) # topK关键词
    max_features = int(sys.argv[3]) #关键词全集数
    bits = int(sys.argv[4]) # 一次学习bits位哈希码
    rand_time = int(sys.argv[5]) # 学习rand_time次
    batch = int(sys.argv[6]) # 一次学习的样本数量
    iters = int(sys.argv[7]) # 一次学习迭代次数
    lambd = float(sys.argv[8]) # 损失参数

    """
    分词/分字,结巴分词/tfidf关键词
    """
    seg_file = fenci(file_name,topK)
    #seg_file = fenzi(file_name)
    #seg_file = "./data/segfile/one_file"
    
    """
    tfidf表示
    """
    (word,tfidf) = Tfidf(seg_file,max_features)
    f = open('./data/argfile/word.utf8', 'w')
    for j in range(len(word)):
        f.write(word[j] + "\n")

    f.close()
    B = BCSH7(tfidf,bits,rand_time,batch,iter,lambd,word)
    print(B.shape)    
    B[B < 0] = 0
    B = B.astype(int)
    f = open('./data/argfile/hashcode.utf8', 'w')
    for i in range(B.shape[1]):
        B_str = ""
        for j in range(B.shape[0]):
            B_str +=str(B[j,i])
        f.write(str(B_str ) + '\t' + str(i) + '\n')
    f.close()

def SemHash_ITSH():
    file_name = sys.argv[1] # 训练集文件
    topK = int(sys.argv[2]) # topK关键词
    max_features = int(sys.argv[3]) #最大特征集合数
    bits = int(sys.argv[4]) # 哈希码长度
    iters = int(sys.argv[5]) # 一次学习迭代次数
    task_name = sys.argv[6] #参数和数据保存路径
    
    task_dir = './data/' + task_name
    
    #tfidf = load_sparse(file_name) 直接加载稀疏表示数据作为输入数据

    """
    分词/分字, 结巴分词/tfidf关键词
    """
    seg_file = fenci(file_name, topK, False, task_dir) #返回分词后的文件
    #seg_file = fenzi(file_name, './data/segfile_tmp')
    #seg_file = "./data/segfile_wiki/one_file"
    
    #tfidf表示
    tfidf = Tfidf(seg_file, max_features, task_dir)

    B = ITSH(tfidf, bits, iters, task_dir)

    #eval_retrieval(B.transpose(), label.transpose(), top_n=100)

def SemHash_BCTH_paper():
    file_name = sys.argv[1] # 训练集文件
    topK = int(sys.argv[2]) # topK关键词
    max_features = int(sys.argv[3]) #最大特征集合数
    bits = int(sys.argv[4]) # 哈希码长度
    iters = int(sys.argv[5]) # 一次学习迭代次数
    task_name = sys.argv[6] #参数和数据保存路径
    
    task_dir = './data/' + task_name

    ##paper实验
    #arg = scipy.io.loadmat('./data/20ng.mat')
    #tfidf = arg["X"]
    #label = arg["Y"]
    #label = np.squeeze(label,axis=0)
    #print(label.shape)
    #print(label)
    #B = ITSH(tfidf, bits, iters)
    #eval_retrieval(B.transpose(), label.transpose(), top_n=100)
    #exit()

    ##哈希表示
    """
    分词/分字,结巴分词/tfidf关键词
    """
    seg_file = fenci(file_name, topK, False, task_dir)
    #seg_file = fenzi(file_name)
    #seg_file = "./data/segfile_wiki/one_file"
    
    #tfidf表示
    tfidf = Tfidf(seg_file, max_features, task_dir)
    
    #label, B, X = init_B("./data/paper_data/clue/inews/train.txt","inews") 
    #label_test, B_test, X_test = init_B("./data/paper_data/clue/inews/test.txt","inews") 
    #label_dev, B_dev, X_dev = init_B("./data/paper_data/clue/inews/dev.txt","inews") 

    label, B, X = init_B("./data/paper_data/clue/tnews/toutiao_category_train.txt","tnews") 
    label_dev, B_dev, X_dev = init_B("./data/paper_data/clue/tnews/toutiao_category_dev.txt","tnews") 
    label_test, B_test, X_test = init_B("./data/paper_data/clue/tnews/toutiao_category_test.txt","tnews") 

    #label, B, X = init_B("./data/paper_data/clue/thucnews/train.txt","thucnews") 
    #label_test, B_test, X_test = init_B("./data/paper_data/clue/thucnews/test.txt","thucnews") 
    #label_dev, B_dev, X_dev = init_B("./data/paper_data/clue/thucnews/dev.txt","thucnews") 

    B = BCSH_paper(B, X, label, B_test, X_test, label_test, B_dev, X_dev, label_dev, bits, iters)

    #eval_retrieval(B.transpose(), label.transpose(), top_n=100)

if __name__ == "__main__": 
    SemHash_ITSH() #最新稳定版
