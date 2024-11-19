#########################################################################
# File Name: demo.py
# Author: yingwenjie
# mail: yingwenjie@baidu.com
# Created Time: Wed 06 Nov 2024 08:40:28 PM CST
#########################################################################
import sys
from demo.demo_cluster import *
from demo.demo_doc_code import *
from demo.demo_doc_sim import *
from demo.demo_new_docs import *
from demo.demo_show_model import *

if __name__ == '__main__':
    task_name = sys.argv[1]
    demo_name = sys.argv[2]
    if demo_name == 'demo_cluster':
        demo_cluster(task_name)
    if demo_name == 'demo_doc_code':
        demo_doc_code(task_name)
    if demo_name == 'demo_doc_sim':
        demo_doc_sim(task_name)
    if demo_name == 'demo_new_docs':
        demo_new_docs(task_name)
    if demo_name == 'demo_show_model':
        demo_show_model(task_name)
