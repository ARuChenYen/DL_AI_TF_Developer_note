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
        這兩者最大的不同在於.from_tensor_slices，會沿著資料第一個維度切片，因此會產生降維的效果
        如果資料shape為(2,3)，經過slices後每個切片會變成shape(3,)
        舉例來說，[[1,2,3], [4,5,6]]，shape為(2,3)
        但是slice後會變成一個tf可迭代的物件。
    
    c.  以上會建立一個TensorDataset。然後迭代出來的會是<class 'tensorflow.python.framework.ops.EagerTensor'>
        直接print會出現 tf.Tensor(1, shape=(), dtype=int32)
        如果用f-string或者.numpy()方法會實際的值打出來
    
    d.  如果資料都已經寫進記憶體了，那麼用.from_tensor_slices()，是最方便且快速轉換成tf.tensor物件
    



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
        print(element.numpy())
        print(type(element))
        print(element)
        
    
    for element2 in tensor_data_from_tensor:
        print(f'from_tensors_iter : {element2}')
        print(element2.numpy())
        print(type(element2))
        print(element2)
       

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

dataset_from(tr_list0)
dataset_from(tr_list1)
dataset_from(tr_list2)

###############################################

def batch_test(data_source, batch_size, padded_shapes, drop_remainder):
    
    data_source = tf.data.Dataset.from_tensor_slices(data_source)
    bat0_with_padding_shape = data_source.padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=drop_remainder)
    bat0_without_padding_shape = data_source.padded_batch(batch_size, drop_remainder=drop_remainder)
    bat1 = data_source.batch(batch_size, drop_remainder=drop_remainder)
    
    print('-----shape of bat0_with_padding_shape-----')
    print(bat0_with_padding_shape.numpy().shape)
    print('-----shape of bat0_without_padding_shape-----')
    print(bat0_without_padding_shape.numpy().shape)
    print('-----shape of bat1 (.batch)-----')
    print(bat1.numpy().shape)

    
    print('-----bat0_with_padding_shape---')
    for element in bat0_with_padding_shape.as_numpy_iterator():
        print(element)
        
    print('-----bat0_without_padding_shape---')
    for element in bat0_without_padding_shape.as_numpy_iterator():
        print(element)
        
    print('-----bat1 (.batch)---')  
    for element in bat1.as_numpy_iterator():
        print(element)
        

tr1 = [1,2,3,]
tr2 = [[1,2,3],[4,5,6]]
tr3 = [[1,2],[3,4,5,6],[7,8,9]]

batch_size = 4
padded_shapes = 4
drop_remainder = False

batch_test(tr1, batch_size, padded_shapes, drop_remainder)