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

2.  Data source
    首先第一步要先把data source弄好，用以下兩種方法可以把存在記憶體中的資料導入
    a.  tf.data.Dataset.from_tensor_slices()
    b.  tf.data.Dataset.from_tensors()
    這兩者最大的不同在於.from_tensor_slices，會沿著資料第一個維度切片，因此會產生降維
    如果資料shape為(2,3)，經過slices後每個切片會變成shape(3,)
    舉例來說，[[1,2,3], [4,5,6]]，shape為(2,3)
    但是slice後會變成一個tf可迭代的物件。



'''

import tensorflow as tf
import numpy as np

def dataset_from(data_source):

    tensor_data_from_slice = tf.data.Dataset.from_tensor_slices(data_source)
    tensor_data_from_tensor = tf.data.Dataset.from_tensors(data_source)
    print(f'Original Data : {data_source}')
    print(f'from_tensor_slices : {tensor_data_from_slice}')
    print(f'from_tensors : {tensor_data_from_tensor}')

    for element in tensor_data_from_slice:
        print(f'from_tensor_slices_iter : {element}')
        print(type(element))
        print(element.numpy())
    
    for element2 in tensor_data_from_tensor:
        print(f'from_tensors_iter : {element2}')
        print(type(element2))
        print(element2.numpy())

tr_list0 = [1,2,3]
tr_list1 = [
    [1,2,3], 
    [4,5,6],
]

tr_list2 = [
    [1,2],
    [3,4],
    [5,6],
]
tr_list3 = [
    {
        'a':[1,2], 'b':[3,4]
    }
]

# nd_tr_list0 = np.array(tr_list0)
# nd_tr_list1 = np.array(tr_list1)

dataset_from(tr_list0)
dataset_from(tr_list1)
dataset_from(tr_list2)
