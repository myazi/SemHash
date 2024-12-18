# -*- coding: utf-8 -*-
"""

@author: yingwenjie

"""

import os
import sys
import string
import numpy as np
import scipy.io
from utils.utils import *

def demo_doc_code(task_name):
    task_dir = "./data/" + task_name
    docs = [] 
    with open(task_dir +  "/seg_file_orgin") as f:
        for line in f:
            docs.append(list(line.strip('\n').split(',')))    
    print("docs num: " + str(len(docs)))
    arg = scipy.io.loadmat(task_dir + "/arg.mat")
    B = arg['B_index']
    bits = arg['logPX1_B1'].shape[0]
    B[B < 0] = 0
    for i in range(len(docs)):
        codes = index2hash(B[0, i], bits)
        print(str(codes) + ':  ' + "".join(docs[i]))

