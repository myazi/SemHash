#########################################################################
# File Name: tesh.sh
# Author: yingwenjie
# mail: yingwenjie@baidu.com
# Created Time: 2022年08月16日 星期二 20时15分03秒
#########################################################################
#!/bin/bash

if [ -n "$1" ]
then
    task_name=$1
else
    task_name="tmp"
fi

rm -rf "./data/${task_name}"
mkdir "./data/${task_name}"

##训练
/root/anaconda3/envs/pytorch/bin/python3 SemHash.py ./data/title_all_sort 10 50000 64 2 $task_name ## 训练样本，关键词数，词表大小，哈希码长度，迭代次数，任务名
#/root/anaconda3/envs/pytorch/bin/python3 SemHash.py ./data/wiki_sample 20 50000 64 10 $task_name

##demo
demo_name="demo_doc_code"
/root/anaconda3/envs/pytorch/bin/python3 demo.py $task_name $demo_name > ./data/${task_name}_${demo_name}
