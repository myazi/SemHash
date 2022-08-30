#########################################################################
# File Name: tesh.sh
# Author: yingwenjie
# mail: yingwenjie@baidu.com
# Created Time: 2022年08月16日 星期二 20时15分03秒
#########################################################################
#!/bin/bash


#/root/anaconda3/envs/pytorch/bin/python3 demo_cluster.py $task_name > ss

if [ -n "$1" ]
then
    task_name=$1
else
    task_name="tmp"
fi

rm -rf "./data/${task_name}"
mkdir "./data/${task_name}"

/root/anaconda3/envs/pytorch/bin/python3 SemHash.py ./data/query_all 10 10000 32  5 $task_name

#/root/anaconda3/envs/pytorch/bin/python3 SemHash8.py ./data/wiki_sample 20 50000 32  5

#/root/anaconda3/envs/pytorch/bin/python3 SemHash_paper.py ./data/query_all 10 10000 32  10 
