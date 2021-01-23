'''
1.  之前的方法
    課程進行到這邊出現了與之前不同call api的方法
    之前的方法如下
    這邊的意思是「建立一個線性模型，裡面有一層，含有64個神經元，激發函數為relu，輸入的型態為(300,300,3)」
'''

import tensorflow as tf

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(64,activation='relu', input_shape=(300,300,3))
    ]
)

'''
https://keras.io/guides/functional_api/
2.  kreas文檔中把每一步驟拆開
    a.  建立輸入層，型態是(300,300,3)
    b.  建立一層神經層，有64個神經元
    c.  將input輸入後得到輸出的結果 x
    d.  將上一層的x再當成下一層(dense(10))的輸入，進入新的一層後得到x
        寫法上就是在 x = layers.Dense(10, activation='relu')(x)
        上一層的輸入放在最後，輸出放前面
    e.  最後將x當成上一層的輸入，輸出為output

3.  建立model
    a.  用model = kreas.Model(inputs=輸入層, outputs=輸出層, name=model名稱)
    
4.  印出模型詳細內容
    keras.utils.plot_model(model, "模型圖片名稱.png", show_shpae='True')
    show_shape決定要不要把每層的輸出輸入印出來

'''

from tensorflow import keras
from tensorflow.keras import layers

input = keras.Input(shape=(300,300,3))
dense = layers.Dense(64, activation='relu',)
x = dense(input)
x = layers.Dense(64, activation='relu',)(x)
output = layers.Dense(10)(x)

model = keras.Model(inputs=input, outputs=output, name='xxxmodel')
keras.utils.plot_model(model, 'test_nmodel.png', show_shapes=True)
