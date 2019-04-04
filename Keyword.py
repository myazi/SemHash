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
#获取文件列表（该目录下放着100份文档）
def getFilelist(argv) :
    path = argv[1]
    filelist = []
    files = os.listdir(path)
    for f in files :
        if(f[0] == '.') :
            pass
        else :
            filelist.append(f)
    return filelist,path
    
def stopwordslist(filename):
    stopword = [line.strip() for line in open(filename,'r',encoding='utf-8').readlines()]
    return stopword
    
#对文档进行分词处???
def fenci(allfile,path) :
    #保存分词结果的目???
    sFilePath = './segfile'
    if not os.path.exists(sFilePath) : 
        os.mkdir(sFilePath)
    #读取文档
    stopwords = stopwordslist('stopwords.utf8')
    #stopwords = ""
    for ff in allfile:
      f = open(path+"/"+ff,'r+',encoding="UTF-8")
      file_list = f.read()
      f.close()
      print ("Using jieba on " + ff)

      file_list = re.sub('[A-Za-z0-9\!\%\[\]]',"",file_list)
      #对文档进行分词处理，采用默认模式
      #seg_list = jieba.cut(file_list,cut_all=True)

      tfidf = analyse.extract_tags
      seg_list = tfidf(file_list, topK=100)
      #print(seg_list)      
      #对空格，换行符进行处???
      result = []
      for seg in seg_list:
          seg = ''.join(seg.split())
          if (seg != '' and seg != "\n" and seg != "\n\n" and seg not in stopwords):
              result.append(seg)

      #将分词后的结果用空格隔开，保存至本地。比???我来到北京清华大???，分词结果写入为??????来到 北京 清华大学"
      f = open(sFilePath+"/"+ff,"w+")
      f.write(' '.join(result))
      f.close()
    return sFilePath
#读取100份已分词好的文档，进行TF-IDF计算
def Tfidf(path,filelist,max_feat=10000) :
    
    corpus = []  #存取100份文档的分词结果
    for ff in filelist :
        fname = path +"/"+ ff
        f = open(fname,'r+')
        content = f.read()
        f.close()
        corpus.append(content)    

    vectorizer = CountVectorizer(max_features=max_feat)    
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    
    word = vectorizer.get_feature_names() #所有文本的关键???
    weight = tfidf.toarray()              #对应的tfidf矩阵
   
    #weight[weight >0 ] = 1
    #weight[weight == 0] = 0

    sFilePath = './tfidffile'
    if not os.path.exists(sFilePath) : 
        os.mkdir(sFilePath)

    # 这里将每份文档词语的TF-IDF写入tfidffile文件夹中保存
    for i in range(len(weight)) :
        f = open(sFilePath+'/'+filelist[i],'w+')
        
        for j in range(len(word)) :
            f.write(word[j]+"    "+str(weight[i][j])+"\n")
        f.close()
    
    return word,weight

