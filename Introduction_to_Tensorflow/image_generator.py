'''
一、ImageGenerator
    ImageGenerator 可以指向一個目錄(directory)
    然後子目錄中分成training與testing
    在其中以資料夾分成不同的種類的圖片，它就會自動label
'''
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageGenerator
train_datagen = ImageGenerator(rescale=1/.255)
# 實例化ImageGenerator，並且正規化

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(300,300),
    batch_size=128,
    class_mode='binary',
)

test_datagen = ImageGenerator(rescale=1/.255)
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(300,300),
    batch_size=32,
    class_mode='binary',
)

'''
二、用ImageGenerator設定訓練與測試資料
1.  用flow_from_directory這個方法，從資料夾讀取訓練資料與測試資料
    並且給予資料的其他細節
    train_dir, validation_dir是指定訓練與測試資料的路徑
    ex: '/tmp/horse_or_human/', '/tmp/horse_or_human_validation/'
    記得要指定到訓練或者測試的資料夾位址
    訓練或者測試資料夾中就包含了用以作為label的類別資料夾，裡面就是圖片了

2.  target_size是圖片的size會調整成300x300，輸入的圖片都要一樣大
    有這行就可以自動幫忙進行調整，而不用自己前處理

3.  batch_size，訓練與資料會批次進行，而不是一個一個訓練
    會更有效率，但怎麼計算batch_size大小目前超出範圍
    設定128(32)代表每次會有128個檔案一起訓練(測試)

4.  class_mode是分類的模型，範例因為是人跟馬兩類，因此用binary
    但當然有其他不同種類的分類器
    以上就是使用ImageGenerator產生一個訓練資料的步驟

5.  調整大小(例如改成150x150)或許會改變訓練的速度，但也可能造成一些誤判
    要如何拿捏訓練會需要好好設計
'''

'''
三、建立CNN網路
    跟之前類似只是有幾處不一樣
1.  前面做了三次卷積，讓size從300x300變成最後35x35，卷積的數量當然可以再調整
2.  input_size最後一項變成3，因為這是RGB的彩色圖片
3.  最後輸出時只有一個神經元，並且用sigmoid函數
    因為sigmoid函數特性相當適合用來做binary分類(輸出0或1)
    所以用一顆神經元配合sigmoid函數會更有效率。
    當然一樣可以用之前兩顆神經元配上softmax函數來做
4.  卷積的時候一開始濾波器設的比較少，後來越來越大
    我認為這可能時因為圖片大小造成的，一開始pixel很大
    如果設太大那麼要算很久
'''

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            16, (3,3), activation='relu', input_shape=(300,300,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu',),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu',),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid',),
    ]
)

'''
四、Compile Model
1.  loss部分使用binary_corssentropy
    因為是二次元分類。
2.  optimizer一樣可以用adam，這邊範例選用RMSprop
    並且設定learning rate為0.01
'''

from tensorflow.keras.optimizers import RMSprop
model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=0.01),
    metrics=['acc']
)

'''
五、訓練模型
    這邊跟之前有較大的不同
1.  使用model.fit_generator()，因為使用ImageGenerator調用資料
2.  放入前面設定的train_generator在第一項
3.  steps_per_epoch跟之前的batch_size與總共的訓練資料密切相關
    因為前面是batch_size為128而且總資料量為1024
    因此steps_per_epoch = 1024/128 = 8
    也就是每次完整訓練一次(一個epoch)，需要幾批(steps)
4.  epochs就是訓練次數
5.  verbose設定每次訓練時要產生多少訊息，設定在2會得到比較少的動畫
6.  validation_data要指定測試的資料，一樣指向先前設定好的validation_generator
7.  validation_step一樣要算走完一個epoch要幾批，之前設定batch為32
    另外測試資料大小為256，因此要8批
'''

model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=2,
    validation_data=validation_generator,
    validation_steps=8,
)

'''
六、預測資料
    當訓練好model後，會想要預測看看資料，以下就是一些程式
1.  前面三行是colab特有的，讓我們可以上傳圖片
2.  np.expand_dim，沿著axis=0新增資料
3.  np.vstack則是沿著垂直方向推疊
    重點要確定輸入資料與模型的資料型態要一致
'''

from google.colab import files
from keras.preprocessing import image
import numpy as np

upload = files.upload()
for fn in upload.keys():
    path = 'content' + fn
    img = image.load_img(path, target_size=(300,300))
    x = image.img_to_array(img)
    x = np.expand_dim(x, axis=0)
    
    images = np.vstack([x])
    classes = model.predict(images,batch_size=10)

    print(classes[0])
    if classes[0]>0.5:
        print(fn + 'is human')
    else:
        print(fn + 'is horse')



