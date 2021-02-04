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

ref:
https://www.tensorflow.org/guide/data

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

'''
3.  padded_batch
    這個方法是把許多連續的元素轉換成一個填充過並且批次處理的數據集
    用padded_batch(batch_size, padded_shapes, padded_values, drop_remainder)
    用as_numpy_iterator轉成可迭代的numpy物件
    a.  batch_size:每次批次處理的量 
        例如 [[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]]
        如果batch_size設為2。那麼用for-loop跑出來的會是[[1,2,3],[4,5,6]]，[[7,8,9],[10,11,12]]
        每批兩個元素。
    b.  drop_remainder: True or Flase，也就是上面批次處理到最後會有不完整的元素
        決定要不要捨棄掉不用，上面的範例就會把最後的[13,14,15]捨棄掉
    c.  padded_shapes: 填充的維度。例如上面元素是3維，設成5的話就會填充到5維
    d.  padding_valuse: 填充值。看要用甚麼值來填充
    
ref:
https://www.tensorflow.org/api_docs/python/tf/data/Dataset#padded_batch

'''

def batch_test(data_source, batch_size, padded_shapes, drop_remainder):
    
    data_source = tf.data.Dataset.from_tensors(data_source)
    bat0_with_padding_shape = data_source.padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=drop_remainder)
    bat0_without_padding_shape = data_source.padded_batch(batch_size, drop_remainder=drop_remainder)
    bat1 = data_source.batch(batch_size, drop_remainder=drop_remainder)
    
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
tr3 = [[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]]

batch_size = 4
padded_shapes = 4
drop_remainder = False

batch_test(tr2, batch_size, padded_shapes, drop_remainder)


###############################################
'''
4.  tf.fill(dim,value)
    這個函數是建立一個維度為dim，但是裡面的元素都是純量value的tensor張量
    例如tf.fill([2,2],3)，就會得到一個值為[[3,3],[3,3]]，shape=(2,2)的tf.Tensor

5.  tf.data.map(mapfun, num_parallel_calls=None, deterministic=None)
    map會把tf中的元素經過mapfun處理後形成新的數據集
    num_parallel_calls，deterministic用法較為高階，暫時不深入研究
    
    記得要調用mpa方法，必須是TensorDataseta的物件，若只有tf.tensor會失敗
    若只有tf.tensor，則要改用tf.map_fn(mapfun, element)，用法類似

    a.  mapfun用法
        目前看到都會使用lambda x: function(...)
        例如 ttt.map(lambda x:x+2)，就會把ttt中的所有元素都加上2
        
        較複雜的函數可以用 lambda x: tf.py_function()
        或者lambda x: tf.numpy_function()
        注意py_function接受tf tensor，而numpy_function接受ndarray
        
        因為tensorflow會經過py_function把函數轉換為圖形來處理
        所以也不能隨便用其他方法在map中調用函數，並且用以上兩種方法都會掉效能
        
ref: 
https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map
https://tensorflow.juejin.im/programmers_guide/datasets.html#toc-10
(optional for deterministic) https://jackd.github.io/posts/deterministic-tf-part-2/
(optional for num_parallel_calls) https://www.tensorflow.org/guide/data_performance
(optional for num_parallel_calls) https://tensorflow.juejin.im/performance/datasets_performance.html
'''

ttr1 = tf.fill([2,2],3)
# ttr1 <tf.Tensor: shape=(2, 2), dtype=int32, numpy=array([[3, 3], [3, 3]])>

# method1，用tf.data.map()
data_ttr1 = tf.data.Dataset.from_tensor_slices(ttr1)
map_data_ttr1 = data_ttr1.map(lambda x:x+2)

# method2，用tf.map_fn()
mapfn_ttr1 = tf.map_fn(lambda x:x+2, ttr1)

# 這兩者輸出都一樣 [array([5, 5]), array([5, 5])]
print(list(map_data_ttr1.as_numpy_iterator()))
print(list(mapfn_ttr1.numpy()))

'''
6.  shuffle用法
    在課程中出現了.shuffle的用法，但是並沒有細部解釋。
    最簡單的理解為，亂數排列
    指令為 tf.shuffle(buffer_size, seed=None, reshuffle_each_iteration=None)
    
    1.  buffer_size
        他會從dataset中，從頭開始取到buffer_size的數量，然後再從中亂數排列
        如果buffer_size中的資料移除，會從原本dataset中buffer_size+1的順位開始填補資料
        
        例如現在我有10000筆dataset，然後buffer_size設為1000
        就會從dataset的第1到1000筆資料取為buffer，buffer不夠時會從dataset第1001筆依序填補至buffer中
    
    2.  注意到原始dataset的隨機性
        因為shuffle是從原始dataset中依序取出作為buffer
        所以如果原始資料分布不平均，會間接導致shuffle並沒有真正的shuffle
        
        例如我有個狗跟貓的數據集
        1-10000為貓的圖片，10001-20000為狗的圖片
        buffer_size為3000。
        那麼在第一批shuffle時看到的只會有貓的圖片，直到第四批才會有狗跟貓的混合
        這是因為tensorflow已經先假定了原始dataset是隨機的
        但如果buffer_size比或者跟原始dataset一樣大，就沒有問題 (前提是記憶體不要爆掉)
        
    3.  以下就有示範，先產生一個0-19，共20個依序排列的tf tensor
        然後設定不同的buffer_size，看看這個對於亂數的影響

https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle
https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
https://zhuanlan.zhihu.com/p/42417456
'''

ori_data = tf.data.Dataset.range(20)
sh0 = ori_data.shuffle(20)
sh1 = ori_data.shuffle(10)
sh2 = ori_data.shuffle(5)
sh3 = ori_data.shuffle(2)
print(list(sh0.as_numpy_iterator()))
print(list(sh1.as_numpy_iterator()))
print(list(sh2.as_numpy_iterator()))
print(list(sh3.as_numpy_iterator()))