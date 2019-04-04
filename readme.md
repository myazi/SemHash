### 文本语义哈希

​	语义哈希（**SemHash**）是一种文本表示方法。同主题模型一样，语义哈希是一个无监督的概率生成模型；不同之处在于，语义哈希采用低维二进制码（64bits）表示文本。用于解决大数据下，近邻查找面临的存储开销大，计算复杂度高，维度灾难等问题。由于采用低维二进制表示，向量空间的内积操作被二进制空间的异或操作取代，显著提高查询效率。该方法的样本时间复杂度为线性，适用于大规模训练数据。模型展示部分包括：文本哈希表示，语义概念推断，文本语义匹配，分四个demo文件。

### 代码运行+环境

环境 python3+jieba+sklearn

##### 1 **模型训练**

​			`$ python3 SemHash.py ./train`

注：训练集可自行构建，下载地址见 链接: https://pan.baidu.com/s/1Swmy23Xru1Yn65Uq1Jrafg 提取码: qmtq 

##### 2 **demo展示**

**文本哈希表示**	: 输出训练集中所有文档的哈希表示

​			`$ python3 hash_code_demo.py`

**语义概念推理**	: 展示每一位哈希码下词的概率分布(topK)

​			`$ python3 show_model_demo.py`

**文本语义匹配** : 展示训练集中每一个文档的相似文档(topK)，给定一个新的文档查找与之相似的文档(topK)

​			`$ python3 doc_sim_demo.py`

​			`$ python3 new_doc_demo.py ./test/红楼梦魇.txt`

### 训练数据和测试数据

**训练集**：（共154篇文档，每篇文档在10000字以上）

​	1 四大名著：红楼梦，三国演义，西游记，水浒传（根据各自长度分成若干子集，红楼梦每十章分为一篇文档，共12篇）

​	2 武侠：射雕英雄传，神雕侠侣，倚天屠龙记，笑傲江湖，鹿鼎记（射雕英雄传分为4篇文档）

​	3 论述：资本论，战争论，富国论（资本论分为6篇）

​	4 恐怖：盗墓笔记，鬼吹灯 （盗墓笔记分为8篇）

​	5 小黄：十本不同来源书籍

​	6 其他：平凡的世界，明朝那些事，金瓶梅，西厢记，飘，乔布斯传

**测试集**：（共7篇）

​	红楼梦魇，天龙八部，三国史话，货币战争，经济学原理，隋唐英雄传，万历十五年，小黄书

### 结果展示

**训练集中文档之间的相似性排序（top20）**

![sim2](C:\Users\86923\Desktop\SemHash\png\sim2.png)

![sim1](C:\Users\86923\Desktop\SemHash\png\sim1.png)



**测试集下新文档与数据库中相似文档排序top20**

![Image 003](C:\Users\86923\Desktop\SemHash\png\Image 003.png)

![Image 001](C:\Users\86923\Desktop\SemHash\png\Image 001.png)

![Image 018](C:\Users\86923\Desktop\SemHash\png\Image 018.png)

![Image 002](C:\Users\86923\Desktop\SemHash\png\Image 002.png)

![Image 017](C:\Users\86923\Desktop\SemHash\png\Image 017.png)

![Image 016](C:\Users\86923\Desktop\SemHash\png\Image 016.png)

![Image 014](C:\Users\86923\Desktop\SemHash\png\Image 014.png)

![Image 015](C:\Users\86923\Desktop\SemHash\png\Image 015.png)