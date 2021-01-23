'''
一、Transfer Learning
    很重要使用tensor flow的技巧，不用自己訓練而使用別人已經花很久或很多資料訓練的模型
    並以此為出發點訓練自己的資料。

1.  可以用lock把別人的模型鎖住，而不用全部重新訓練，並且用別人的模型提取特徵，然後用自己的DNN訓練
2.  也不用全部鎖住，只鎖住部分的，然後重新訓練模型中較底層的CNN，因為別人的資料可能跟自己的有很大的差距
    這需要try and error來發現較好的組合
'''

'''
二、使用pre_train model
1.  初始的import項目
    因為要調整layer跟model的使用時機與方式
    因此要從tensorflow.keras import layers與Model

2.  import InceptionV3
    在keras.applications提供了帶有預訓練的深度學習模型
    會根據配置文件設置圖像數據格式的建構模型

3.  設定pre_train的參數
    實例化InceptionV3時，要依照自己的資料格式與類型設定參數
    1. input_shape，資料的輸入形式
    2. include_top=False，不使用預載的全連接層(fully connect layer)
       所謂的全連接層就是接收與輸出都會連接到全部的neuron。
       預載的模型中在頂層有全連接層，但我們想要直接進入convolution，因此設定忽略
    3. weights=None，不只用預設的權重，而是要使用其他地方的參數

3.  載入別人的模型的權重
    用.load_weights讀入檔案
    pre_train_model.load_weights(local_weight_file)

4.  用for-loop迭代整個pre_train_model，然後把每層layer設定trainable=false
    layers.trainable = False

'''

import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weight_file = '...'

pre_train_model = InceptionV3(
    input_shape=(150,150,3),
    include_top=False,
    weights=None,
)

pre_train_model.load_weights(local_weight_file)

for layers in pre_train_model.layers:
    layers.trainable = False

'''
三、設定自己的底層DNN
1.  選取最後一層
    每一層都有名稱，用model.summary()查看，並且用.get_layer('layer_名稱') 取用該層
    last_layer = pre_train_model.get_layer('mixed7')

2.  選取最後一層的輸出。
    只選最後一層還不夠，因為包含太多資訊，用.output來輸出結果，並作為我們DNN網路的輸入
    last_output = last_layer.output

3.  設定layers
    基本上使之前的方法一樣，只是用一些不同的方法使用layers這個api
    這邊的寫法跟之前的不一樣，看keras文檔會更清楚
    詳細的寫在lesson3_extra1.py中。
    簡單來說，上一層的輸入會放在本層最後當成輸入

4.  建立model
    一樣看文檔，方法如下
    model(輸入層, 輸出層, name='xxx')

5.  其他的步驟就跟之前一樣，建立imagegenerator...
6.  但是這樣建立後會發現驗證資料中產生overfitting的情況

'''
last_layer = pre_train_model.get_layer('mixed7')
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_train_model.input, x)

model.compile(
    optimizer=RMSprop(lr=0.01),
    loss='binary_crossentropy',
    metrics=['acc'],
)



