'''
六、 順序
    字的順序很重要，順序錯了可能句子會變得一點意義也沒有
    方法可以用RNN, LSTM

1.  LSTM
    有額外的pipline，裡面有單元格儲存上下文(cell state)，可以保存前後內容的關聯性
    並且因為是雙向的，所以後出現的內容也可能會影響到前面出現的內容。
    另外建立管道也是避免直接影響到彼此前後文的內容。
    
    a.  指令 tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))
        用bidirectional, 因為這是雙向的。也因此輸出shape會是128 (64*2)
    
    b.  LSTM可以疊LSTM，但是在前一層要加上(return_sequence=True)，這樣可以確保前一層輸出匹配到後一層輸入
    
    c.  用了LSTM，對比沒有使用LSTM，移除了Flatten()與GlobalAveragePooling1d()
        等於直接從embedding層->LSTM層->其他Dense層
    
    d.  一樣要注意overfitting的問題，訓練過程中，如果訓練loss下降但驗證的loss持續上升
        那麼很有可能已經overfitting了
        
    e.  GRU的用法幾乎跟LSTM一樣，但一樣會有overfitting的問題
    


2.  其他地方沒有修改，就把lesson3_subword_pre_token中的model部分代換成這邊的code就可以跑LSTM了
    結果可以發現兩層的LSTM在accuracy，loss上都會比一層的還要平滑，較不會出現劇烈的上下震盪

3.  使用CNN進行
    跟之前圖片類似，只是把filter的形狀改成5(本案例中)
    tf.keras.layers.Conv1D(128, 5)
    
    a.  本範例中，使用了128個filter，每個濾波器運算時都是以五個單字一組進行
        原本max_length為100，但是經過Conv1D後變成96，因為會去除頭尾兩個字
        但是去除單字的多少，會跟濾波器取多少個字為一組有關。
        本案例中是設定5，所以頭尾各去除2，如果設定更大，那麼頭尾會去除更多

'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


vocab_size = 10000
max_length = 100

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64,input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

model.summary()

model2 = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64,input_length=max_length),
    tf.keras.layers.Conv1D(128, 5, activation='relu',),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

model2.summary()

model3 = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64,input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

model3.summary()