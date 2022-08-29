# -*- coding: utf-8 -*-
"""

@author: yingwenjie

"""

import os
import jieba
import jieba.posseg as pseg
from jieba import analyse
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re

def stopwordslist(filename):
    stopword = [line.strip() for line in open(filename,'r',encoding='utf-8').readlines()]
    return stopword

def vocab(filename):
    voc = [line.strip() for line in open(filename,'r',encoding='utf-8').readlines()]
    return voc
    
#对文档进行分词处
def fenci(file_name, topK=10, stopwords_flag = True, seg_dir = "./data/tmp"):
    stopwords = ""
    if stopwords_flag:
        stopwords = stopwordslist('./data/stopwords.utf8')
    if not os.path.exists(seg_dir) : 
        os.mkdir(seg_dir)
    wf = open(seg_dir + "/seg_file","w")
    wf_orgin = open(seg_dir + "/seg_file_orgin","w")
    i = 0
    with open(file_name) as f:
      for line in f:
         line_list = line.strip().split('\t')
         sample = line.strip()
         """
         if "tnews" in file_name:
             nid,label,label_text,content1,content2 = line.strip('\n').split('_!_')#tnews
         if "thucnews" in file_name:
             label,label_text,content1,content2 = line.strip('\n').split('_!_')#thucnews
         if "inews" in file_name:
             label,pid,content1,content2 = line.strip('\n').split('_!_')#inews
         sample = content1  + "\t" + content2
         """ 
         sample = re.sub('[A-Za-z0-9\!\%\[\]\"\:\/\?\.\_\-\=]'," ",sample)
        
         #对文档进行分词处理，采用默认模式
         #seg_list = jieba.cut(sample, cut_all=False)
         #seg_list = jieba.cut(sample, cut_all=True)
         
         #tfdif提取关键词
         tfidf = analyse.extract_tags
         seg_list = tfidf(sample, topK)

         i += 1
         if i % 100000 == 0:
            print(sample + "\t" + "|".join(seg_list))

         result = []
         for seg in seg_list:
             seg = ''.join(seg.split())
             if (seg != '' and seg != "\n" and seg != "\n\n" and seg not in stopwords):
                result.append(seg)
         if len(result) < 2: continue
         if result:
             res = '|'.join(result) + "\n"
             res_orgin = line.strip('\n') + "\t" + res
             wf.write(res)
             wf_orgin.write(res_orgin)

      wf.close()
      wf_orgin.close()

    return seg_dir + "/seg_file"

#分字
def fenzi(file_name, seg_dir = "./data/tmp") :
    if not os.path.exists(seg_dir) : 
        os.mkdir(seg_dir)

    wf = open(seg_dir + "/seg_file","w")
    wf_orgin = open(seg_dir + "/seg_file_orgin","w")
    i = 0
    with open(file_name) as f:
      for line in f:
         line_list = re.sub('[A-Za-z0-9.\!\%\[\]]',"",line.strip())
         if len(line_list) == 0: continue
         seg_list = list(line_list)
         result = []
         for seg in seg_list:
             seg = ''.join(seg.split())
             if (seg != '' and seg != "\n" and seg != "\n\n"):
                result.append(seg)
         i+=1
         if i % 100000 == 0:
             print(line_list + "|".join(result))
         res = '|'.join(result) + "\n"
         res_orgin = line.strip() + "\t" + res
         wf.write(res)
         wf_orgin.write(res_orgin)
      wf.close()
      wf_orgin.close()
    return seg_tmp + "/seg_file"

#读取已分词好的文档，进行TF-IDF计算
def Tfidf(seg_file, max_feat=5000, arg_dir = "./data/tmp") :
    if not os.path.exists(arg_dir): 
        os.mkdir(arg_dir)
    corpus = []  #存取文档的分词结果
    corpus_seg = []  #存取文档的分词结果
    i = 0
    with open (seg_file) as f:
        for line in f:
            if len(line.strip()) == 0:
               continue
            corpus.append(line.strip().replace('|', ' '))
            corpus_seg.append(line.strip().split("|"))
            i += 1
            if i % 100000 == 0:
                print(str(i) + "\t" + line.strip())
    """
    统计和存储各个关键词出现的次数
    """
    tfidf_word_file = open(arg_dir + "/word_num",'w')
    keys_dict = {}
    for item in corpus_seg:
        for key in item:
            keys_dict.setdefault(key, 0)
            keys_dict[key] += 1
    keys_list =  sorted(keys_dict.items(),key = lambda x:x[1], reverse = True)
    for key, value in keys_list:
        tfidf_word_file.write(str(key) + "\t" + str(value) + "\n")

    #vectorizer = CountVectorizer(max_features=max_feat, analyzer = "char",lowercase=False) #分字    
    vectorizer = CountVectorizer(max_features=max_feat, lowercase=False)    
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    print(tfidf)
    
    word = vectorizer.get_feature_names() #所有文本的关键词
    f = open(arg_dir + '/arg_word.utf8', 'w')
    for j in range(len(word)):
        f.write(word[j] + "\n")
    f.close()
    #weight = tfidf.toarray()              #对应的tfidf矩阵
    #weight[weight >0 ] = 1
    #weight[weight == 0] = 0

    return tfidf 
