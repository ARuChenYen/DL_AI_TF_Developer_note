'''
一、Image Augment
1.  訓練資料太少容易導致overfitting，因此用ImageAugment加強
2.  Image Augment在記憶體中運作，也不會直接修改到原始資料，可以將圖片旋轉或做一些簡單的修改
    讓圖片可以模擬各種情況的狀態來訓練。
3.  在實例化ImageGenerator時設定
4.  一些參數細節
    rotation_range=40，圖片會在0-40度旋轉
    shift_range=0.2，圖片會左右或上下移動20%
    shear_range=0.2，圖片會沿著各方向裁減20%
    zoom_range=0.2，圖片隨機縮放，最大達20%
    horizontal_flip=True，讓圖片水平翻轉
    fill_mode='nearest'，圖片在操作的過程可能丟失掉pixel，因此要填充回來。
    而nearest就是拿最接近的pixel來填充。

5.  不只訓練需要多樣性，測試的資料也要有多樣性
    當然Image Augment並不是解決所有overfitting的問題，但依舊是非常棒的方法

6.  設好image augment後訓練會慢很多，因為處理圖片也需要時間。
'''

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageGenerator

train_datagen = ImageGenerator(
    rescale=1/.255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    )


