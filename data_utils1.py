#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!/usr/bin/env python
# coding: utf-8

import re
import pandas as pd
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizers import Tokenizer

# 定义实体标签
entity_labels = ["有杆泵抽油机", "运行情况", "设备", "故障", "参数特征", "抽油机", "井下抽油泵", "抽油杆", "示功图", "构造特征", "采油特征", "地面设备故障", "井下设备故障", "地面设备运行工况", "井下设备情况", "工作环境工况", "驴头", "减速箱", "连杆", "泵筒", "固定阀", "游动阀", "柱塞", "电动机", "曲柄", "示功图特征", "抽油效率", "支架", "游梁", "中央轴承座", "轴承", "中央轴", "尾轴", "螺栓", "螺帽", "工作现象", "管柱", "油井特征", "故障特征", "特征变化", "齿轮", "齿轮轴承", "油管", "示功图变化", "光杆", "泵阀", "理论示功图", "衬套", "阀球"]


id2label = {i:j for i,j in enumerate(sorted(entity_labels))}
label2id = {j: i for i, j in id2label.items()}
num_labels = len(entity_labels) * 2 + 1 # b i o

vocab_path = 'E:/jupyter/chinese_L-12_H-768_A-12/vocab.txt' 
tokenizer = Tokenizer(vocab_path, do_lower_case=True)
max_len = 150

# 修改后的 load_data 函数
def load_data_modified(data_path, max_len):
    """加载数据并保留单个字符的BIO标注
    单条格式：[['字', 'B-tag'], ['字', 'I-tag'], ...]
    """
    datasets = []
    X = []
    y = []
    
    with open(data_path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip().split()
            if not line or len(line) < 2:
                datasets.append(list(zip(X, y)))
                X = []
                y = []
                continue

            word = line[0]
            tag = line[1]
            X.append(word)
            y.append(tag)

    # 添加最后一个样本
    if X and y:
        datasets.append(list(zip(X, y)))

    return datasets

# 使用修改后的函数处理文件
data_path = 'E:/jupyter/chouyouji data/converted_annotations_bio.txt'
modified_data = load_data_modified(data_path, max_len)

# 打印前几个样本的内容
print(modified_data[:300])


# In[ ]:




