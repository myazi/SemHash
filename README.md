# **语义哈希（Semantic Hash）**
语义哈希是一种离散向量表示方法，通过将数据从原始空间映射到低维二进制(64bits)空间，解决大规模数据下近邻检索面临的存储开销大、计算复杂度高、维度灾难等问题. 

这里我们首次提出BCTH: A Novel Text Hashing Approach via Bayesian Clustering，通过贝叶斯聚类学习文档的二进制表示、同时引入code balance来保证二进制码的质量. 然而，BCTH中code balance限制了算法的性能(难以处理亿级别训练数据). 因此，提出ITSH: Iterative Semantic Hashing，按位(bit)交替迭代学习二进制码，在每一位二进制码学习过程中建模code balance. 相对BCTH，ITSH在算法性能和二进制码质量上有显著提升. 同时，我们发现语义哈希能学习到语义相近的词具有相近的二进制，对此我们也在探索词的哈希表示(word2hash).

### BCTH: A Novel Text Hashing Approach via Bayesian Clustering

![BCTH1](./docs/img/BCTH1.png)

### ITSH: Iterative Semantic Hashing

![ITSH1](./docs/img/ITSH1.png)

![ITSH2](./docs/img/ITSH2.png)

## 环境
python3

sklearn 0.24.2

scipy 1.5.4

jieba 0.42.1

## 数据
**测试数据:** ./data/query_all_sub

**中文wiki数据:** 链接: https://pan.baidu.com/s/1YTvSCA38O9q2Sknqk0gfzg?pwd=h7te 提取码: **h7te**

## 使用
```
sh run.sh

## 训练
python3 SemHash.py ./data/query_all_sub 10 10000 32 5 $task_name ## 训练样本，关键词数，词表大小，哈希码长度，迭代次数，任务名

## 输出文档哈希码
python3 demo_doc_code.py $task_name > ${task_name}_code

## 输出测试文档的相似文档
python3 demo_doc_sim.py $task_name > ${task_name}_sim

## 输出文档的哈希码簇
python3 demo_cluster.py $task_name > ${task_name}_cluster

## 输出词的哈希码、词的相似关系
python3 demo_show_model.py $task_name > ${task_name}_model
```

## 效果
**[10万query测试集 32bits 哈希码聚簇]**
| 哈希码index | query |  哈希码index | query |
| ----------------- | ------ | ----------------- | ------ |
| 2747264068 | 三亚 到底在哪儿 | 2253996783 | 01倚天屠龙记演员表 |
| 2747264068 | 三亚 哪个省 | 2253996783 | 01年倚天屠龙记演员表
| 2747264068 | 三亚 在哪个省 | 2253996783 | 01版倚天屠龙记演员列表
| 2747264068 | 三亚 属于那个省 | 2253996783 | 03倚天屠龙记演员表 |
| 2747264068 | 三亚 崖州机场 | 2253996783 | 03年倚天屠龙记演员表 |
| 3344118251 | 58同城上市什么时间 | 889120837 | 上海浦东新区 机场 |
| 3344118251 | 58同城上市多少年了| 889120837 | 上海浦东新区区号多少 |
| 3344118251 | 58同城上市时间 | 889120837 | 上海浦东新区占地面积 |
| 3344118251 | 58同城什么时候上市的 | 889120837 | 上海浦东新区占地面积多少 |
| 3344118251 | 58同城哪一年上市 | 889120837 | 上海浦东新区占地面积是多大 |
| 3344118251 | 58同城啥时候上市的 | 889120837 | 上海浦东新区哪个机场 |

**[中文wiki训练数据 64bits]**
| 单词              | 相似单词 |
| ----------------- | ----- |
| 一一对应 | 下界:64,不动点:64,乘法:64,代数:64,值域:64,偏序:64,公理:64,加法:64,半群:64,可微:64,可数:64,可测:64,可积:64,同伦:64,基数:64,多项式:64,子集:64,定义域:64,实数:64,就是说:64,展开式:64,序数:64,微分:64,拓扑:64,收敛:64,数列:64,无理数:64,无穷:64,有理数:64,柯西:64,正则:64,测度:64,狄利克:64,空集:64,等价:64,等价关系:64,算子:64,级数:64,阶乘:64,集合:64,非负:64,下述:62,不等式:62,二元:62,交集:62,伯努利:62,伽罗瓦:62,例子:62,全纯:62,单调:62,反例:62,可分:62,可定义:62,同构:62,同调:62,向量场:62,复形:62,子群:62,完备:62,定理:62,导数:62,布尔代数:62,希尔伯特:62,幂级数:62,庞加莱:62,开集:62,引理:62,归纳法:62,当且:62,恒等:62,抽象代数:62,拉普拉斯:62,拓扑学:62,整数:62,映射:62,有界:62,欧几里得:62,正数:62,正整数:62,泊松:62,流形:62,猜想:62,紧致:62,维数:62,若且唯若:62,范数:62,蕴涵:62,论中:62,负数:62,质数:62,连续函数:62,邻域:62,闭包:62,除法:62,非零:62,黎曼:62,一般化:60,一阶:60,上同调:60,乘积:60 |
| 丁烷 | 丁基:64,丙烯:64,丙烷:64,丙酮:64,乙基:64,乙炔:64,乙烯:64,亚胺:64,产率:64,亲核:64,亲电:64,催化剂:64,制取:64,制备:64,加成:64,加成反应:64,化学式:64,叠氮:64,吡啶:64,有机合成:64,溴化:64,烯烃:64,环戊二烯:64,甲苯:64,硝基:64,缩合:64       ,羟醛:64,芳基:64,苯基:64,苯胺:64,试剂:64,路易斯酸:64,重排:64,丁酸:62,中间体:62,乙烷:62,乙酯:62,乙酰:62,乙酸:62,乙醇:62,乙醚:62,二价:62,二氯:62,二甲基:62,五元:62,亚硝酸:62,内酯:62,分子式:62,制得:62,副产物:62,单质:62,卤化:62,卤化物:6       2,卤素:62,同分异构:62,咪唑:62,强碱:62,无机化合物:62,杂环:62,氟化:62,氟化物:62,氢化:62,氢氧化:62,氢氧化物:62,氧化:62,氧化剂:62,氨基:62,氯仿:62,氯化:62,氯化氢:62,氯化物:62,氰酸:62,热分解:62,烯醇:62,烷基:62,烷烃:62,环化:62,甲基:62,甲>       酸:62,甲醇:62,甲醛:62,硝化:62,硝酸:62,硫化:62,硫氰酸:62,硫酸:62,硫醇:62,硼酸:62,碘化:62,磺酸:62,类化合物:62,羟基:62,羧基:62,羧酸:62,羰基:62,苯甲酸:62,衍生物:62,负离子:62,质子化:62,还原:62 |
| 望远镜 | 光谱学:64,哈伯:64,哈勃:64,射电:64,惠更斯:64,类星体:64,脉冲星:64,中子星:62,候选者:62,倾角:62,共振:62,卡西尼:62,吸积:62,太阳黑子:62,干涉仪:62,探测器:62,星体:62,最轻:62,有效温度:62,木卫三:62,木卫二:62,木卫四:62,木星:62,极光:62,水星:62,红外:62,红移:62,行星:62,观测:62,超大:62,重元素:62,闪焰:62,颗卫星:62,云气:60,仪器:60,伽利略:60,低质量:60,侦测器:60,假想:60,光度:60,光源:60,光球:60,光线:60,光谱:60,光谱仪:60,冥王星:60,凌日:60,分光:60,动力学:60,反射:60,可见光:60,周期:60,喷流:60,土卫六:60,土星:60,地球:60,坍缩:60,埃欧:60,塌缩:60,多普勒:60,大到:60,天王星:60,密度:60,射线:60,尘埃:60,尺度:60,巡天:60,并合:60,开尔文:60,引力:60,引力场:60,引力波:60,微小:60,折射:60,抛射:60,排布:60,探测:60,摄动:60,日冕:60,时间尺度:60,暗物质:60,柯伊伯:60,核聚变:60,检测器:60,正电荷:60,母星:60,毫秒:60,波段:60,波长:60,海王星:60,火卫一:60,火星:60,焦耳:60,电浆:60,电离:60,白矮星:       60,盘面:60,真空:60,离心力:60,稀薄:60 |
| 七七事变 | 抗日:62,东北军:60,中央红军:60,中央苏区:60,九一八事变:60,八路军:60,冀东:60,北伐战争:60,华北:60,国共:60,抗日战争:60,救国军:60,新四军:60,日伪:60,易帜:60,汤恩伯:60,派遣军:60,游击战争:60,红一方面军:60,红四方面军:60,苏区:60,解放区:60,解放战争:60,鄂豫皖:60,长征:60,陕北:60,一二八:58,一师:58,东北民主联军:58,中国工农红军:58,义勇队:58,井冈山:58,何应钦:58,关东军:58,兵站:58,冈村:58,军政:58,军长:58,刘伯承:58,剿共:58,剿匪:58,十九路:58,华东野战军:58,南昌起义:58,卫戍:58,参谋:58,参谋总长:58,叶挺:58,司令部:58,合编:58,团团长:58,团部:58,围剿:58,国民革命军:58,大后方:58,大队长:58,宁汉:58,宋哲元:58,师师:58,师师长:58,师长:58,平津:58,归绥:58,徐向前:58,总司令部:58,总指挥:58,抗战:58,抗美援朝:58,抗联:58,排长:58,支队:58,敌后:58,教导团:58,整编:58,旅长:58,晋察冀:58,晋察冀边区:58,暂编:58,淞沪:58,淮海战役:58,混成旅:58,满洲国:58,独立团:58,白崇禧:58,百团大战:58,皖南事变:58,第三军:58,第三师:58,粟裕:58,红三军团:58,红军:58,统一指挥:58,绥靖:58,绥靖区:58,自卫军:58,苏中:58,营长:58,西北军:58,警备:58,警备司令:58 |
| 牙齿 | 下颌:64,前肢:64,嘴部:64,头冠:64,脊椎:64,颈椎:64,颌部:64,骨头:64,鳞甲:64,鼻部:64,上颌:62,上颌骨:62,冠饰:62,前段:62,双足:62,后肢:62,四肢:62,尾椎:62,正模:62,犬齿:62,结节:62,肋骨:62,股骨:62,肩胛骨:62,肱骨:62,胫骨:62,脚掌:62,臼齿:62,镰刀:62,门齿:62,颅骨:62,骨板:62,骨盆:62,上颚:60,下颚:60,修长:60,口器:60,喉部:60,嘴龙:60,四足:60,头部:60,头骨:60,演化出:60,状物:60,眼眶:60,结实:60,耻骨:60,肩带:60,肩部:60,胸部:60,臀部:60,角质:60,躯干:60,长而:60,颈部:60,骨骼:60,鼻孔:60,体型:58,体表:58,前额:58,剑龙:58,后段:58,咀嚼:58,圆点:58,圆锥:58,坚硬:58,复眼:58,大而:58,尖刺:58,尖角:58,尾巴:58,幼体:58,强壮:58,很长:58,恐爪:58,愈合:58,成体:58,扁平:58,手掌:58,易碎:58,标本:58,水族箱:58,爪子:58,甲壳:58,皮毛:58,瞳孔:58,破碎:58,管状:58,红色:58,细小:58,羽毛:58,翅膀:58,背侧:58,脚趾:58,腹侧:58,膨大:58,虹膜:58,蟒蛇:58,表层:58,足部:58 |
| 曹操 | 公孙瓒:64,关羽:64,军粮:64,刘备:64,刘牢之:64,刘璋:64,刘表:64,刘邦:64,吕布:64,吕蒙:64,周勃:64,周瑜:64,夏侯渊:64,大乱:64,姜维:64,孙坚:64,孙权:64,孙策:64,引兵:64,张鲁:64,数万:64,曹仁:64,檄文:64,用兵:64,缺粮:64,萧何:64,董卓:64,袁尚:64,袁术:64,袁绍:64,袁谭:64,诸葛亮:64,谋士:64,赤壁之战:64,轻骑:64,运粮:64,不下:64,之众:64,之策:64,连年:64,邓艾:64,钟会:64,陆逊:64,项羽:64,马超:64,鲁肃:64,三十万:62,三千余:62,东吴:62,中行:62,之兵:62,之势:62,之盟:62,之计:62,二万:62,五千:62,令狐:62,会盟:62,何进:62,余众:62,余党:62,作乱:62,侯景:62,元军:62,元氏:62,兴兵:62,兵变:62,兵权:62,内应:62,军师:62,出城:62,出降:62,击破:62,函谷关:62,刘曜:62,刘毅:62,刘琨:62,刘禅:62,刘聪:62,刘裕:62,前燕:62,助战:62,募兵:62,北宫:62,十万:62,十余:62,十余万:62,单骑:62,南侵:62,南燕:62,卢循:62,厚待:62,发丧:62,司马师:62,司马懿:62,司马昭:62,司马氏:62,司马越:62,司马道子:62,司马颖:62 |
| 一院制 | 两院制:62,小党:62,国民议会:60,欧洲议会:60,立法机关:60,立法机构:60,两院:58,代表制:58,众议院:58,党籍:58,制宪会议:58,执政党:58,绝对多数:58,自民党:58,议会:58,议院:58 |
## 文献
