# -*- coding: utf-8 -*-
"""

@author: yingwenjie

"""

import os
import sys
import string
import numpy as np
import scipy.io

if __name__ == "__main__" :
    task_name = sys.argv[1]
    task_dir = "./data/" + task_name
    docs = [] 
    with open(task_dir +  "/seg_file_orgin") as f:
        for line in f:
            docs.append(list(line.strip('\n').split(',')))    
    print(len(docs))    
    arg = scipy.io.loadmat(task_dir + "/arg.mat")
    B = arg['B']
    B[B < 0] = 0
    for i in range(len(docs)):
        codes = ""
        for j in range(B.shape[0]):
            codes = codes + str(B[j, i])
        print(str(codes) + ':  ' + "".join(docs[i]))

