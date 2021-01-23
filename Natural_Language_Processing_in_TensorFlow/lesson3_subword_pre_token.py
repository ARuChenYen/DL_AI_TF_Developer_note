'''
六、Loss Function
    從Sarcasm的練習3.中跑完會發現測試資料的loss隨著訓練一直上升，accuracy下降
    這時就需要調整

1.  調整超參數
    減少voca_size，max_length，讓詞庫與填充少點，會發現情況有改善
    或者增加embedding維度，等等方法都可以探索對於訓練的影響
'''

'''
七、pre_token，與subword
    使用一些現有的data set，大部分都已經有pre-token了
    以IMDB Review舉例，就已經在subword上有tokeinzer了
    所謂的subword就是從一個完整單字中，擷取部分的字母構成的subword
    但是subword通常是沒有意義的，除非配合上特定的順序。
    因此順序要讓神經網路學會順序是非常重要的事情
    本案例中單純以DNN訓練的話結果很不好。

1.  一樣先import tensorflow_datasets

2.  tokenizer = info.features['text'].encoder
    看看subword的tokenizer

3.  tokenizer還有幾個功能
    a.  print(tokenizer.subwords)
        看看subword的字彙表
    b.  tokenizer.encode(sample_text)
        tokenizer.decode(encoded_sequence)
        可以看看編碼前與解碼後的串列或者文字
        解碼時放入decoder的必須是list，若直接放int會出錯

4.  取出訓練與測試資料
    imdb['train'], imdb['test']
    因為這些資料已經pro-tokenizer了，因此不用像是lesson2那樣自己走一次

5.  建立並訓練神經網路
    這部分跟之前類似，但不要用Flatten()
    因為本案例中不好Flatten，用下去會當掉
    因此改用GlobalAveragePooling1D
    訓練較為耗費時間

'''

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

imdb, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
training_data = imdb['train']
testing_data = imdb['test']

tokenizer = info.features['text'].encoder

print(tokenizer)

sample_text = 'Tensorflow, from basics to mastery.'
sa_encode = tokenizer.encode(sample_text)
sa_decode = tokenizer.decode(sa_encode)

print(f'Origin String:{sample_text}')
print(f'After encode:{sa_encode}')
print(f'After decode:{sa_decode}')

for i in sa_encode:
    decode_word = tokenizer.decode([i])
    print(f'{i} ------> {decode_word}')

embedding_dim = 64
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim,),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ]
)

model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy'],
)

num_epochs = 10
history = model.fit(
    training_data,
    epochs=num_epochs,
    validation_data=testing_data,
)

# Plot result
def plot_result(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val'+string])
    plt.show()

plot_result(history, 'accuracy')
plot_result(history, 'loss')