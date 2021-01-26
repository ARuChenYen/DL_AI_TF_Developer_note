'''
Extra:  tf.data.Dataset
        在影片後的colab中，不僅僅把train_data = imdb['train']取出
        還用了新的方法 .shuffle(buffer_size)，padded _batch()
        這邊如果不去看文檔會完全不知道在做甚麼，甚至每個參數的意義也不清楚
        因此本篇另外先從tensorflow dataset文檔解釋重要的功能
        之後在接回去lesson3_subword_pre_token.py，要如何才能讓資料訓練

1.  tf.data
    用這個api來建立input pipline。會用pipline(流水線)是因為處理的資料量很大
    直接塞進去記憶體會爆掉，因此要建立流水線批次處理資料
    例如本次處理文字時，就需要提取文字，embedding，還要批次組相同長度的sequence
    使用tf.data.dataset可以處理大量資料，並且前處理成tf.data.dataset物件

2.  基本功能
    a.  tf.data.dataset.

'''

import tensorflow as tf
import numpy as np


dataset1 = tf.data.Dataset.from_tensor_slices()
