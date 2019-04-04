# -*- coding: utf-8 -*-
"""

@author: yingwenjie

"""

import os
import sys
import string
import numpy as np
import scipy
from Keyword import *
from BCSH1 import *
from BCSH2 import *
from Normalize import *
from utils import *

if __name__ == "__main__" : 
    
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




