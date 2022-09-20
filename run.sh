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
/root/anaconda3/envs/pytorch/bin/python3 SemHash.py ./data/query_all_sub 10 10000 32 10 $task_name ## 训练样本，关键词数，词表大小，哈希码长度，迭代次数，任务名
#/root/anaconda3/envs/pytorch/bin/python3 SemHash.py ./data/wiki_sample 20 50000 64 10 $task_name

##demo
/root/anaconda3/envs/pytorch/bin/python3 demo_doc_code.py $task_name > ${task_name}_code
/root/anaconda3/envs/pytorch/bin/python3 demo_doc_sim.py $task_name > ${task_name}_sim
/root/anaconda3/envs/pytorch/bin/python3 demo_cluster.py $task_name > ${task_name}_cluster
/root/anaconda3/envs/pytorch/bin/python3 demo_show_model.py $task_name > ${task_name}_model
