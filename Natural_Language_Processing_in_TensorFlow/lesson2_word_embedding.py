'''
四、讓文字編碼含有更多情境
    之前的方式編碼是使用key, value值，並且組成一個完整的矩陣
    但如果要讓一個詞含有更多情緒(sentiment)，就要用word embedding的方法
    類似之前訓練圖片的特徵一樣，word embedding讓字可以得到更多與之相似的字的張量作為特徵

1.  使用build-in的資料先嘗試。導入tensorflow dataset
    import tensorflow_datasets as tfds
    with_info: 連帶這個資料的info都會讀出來
    as_supervised: 設定為True，資料型態會變成一個tuple(input, label)
    如果為False，就會變成跟tf.data.Dataset一樣用dict來保存所有特徵
    
2.  分成訓練與測試資料
    train_data, test_data = imdb['train'], imdb['test']
    這時候的data是PrefetchDataset

3.  用for-loop迭代出sentence跟labels
    這邊比較複雜一點，要參考文檔 https://www.tensorflow.org/guide/tf_numpy
    會有.numpy()是因為tensor有把numpy的一些功能寫進去，所以可以無縫的交互使用
    用了.numpy()就會把tf.tensor轉成實際上的資料讀取出來了。
    如果沒有加上.numpy則依然會是tf.tensor張量

4.  轉成ndarray
    np.array()
    因為要進入神經網路訓練，labels必須是numpy array，因此轉換成ndarray
    剛提取出來時整個labels還是list

5.  接下來就是tokenizer，與之前的類似

'''

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
train_data, test_data = imdb['train'], imdb['test']

train_sentences = []
train_labels = []
test_sentences = []
test_labels = []

for s, l in train_data:
    train_sentences.append(str(s.numpy()))
    train_labels.append(l.numpy())

for s, l in test_data:
    test_sentences.append(str(s.numpy()))
    test_labels.append(l.numpy())

print(type(train_sentences))
print(type(train_labels))

train_labels_ndarray = np.array(train_labels)
test_labels_ndarray = np.array(test_labels)

'''
5.  接續上面，文字tokenizer
    只是本次我們先設定超參數(hyper parameters)，好處是以後只要在一處修改參數就好。
    因為建立詞庫是用訓練資料，因此預期在測試資料的sequence中應該會有一些unseen word

6.  建立神經網路

'''

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_token = '<oov>'
num_epochs = 10

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(train_sentences)
sequence_train_sentences = tokenizer.texts_to_sequences(train_sentences)
padded_s_tr_sentence = pad_sequences(sequence_train_sentences, maxlen=max_length, truncating=trunc_type)

word_index = tokenizer.word_index
sequence_test_sentences = tokenizer.texts_to_sequences(test_sentences)
padded_s_te_sentence = pad_sequences(sequence_test_sentences, maxlen=max_length, truncating=trunc_type)


'''
6.  建立神經網路
    最重要的是第一層加了一層embedding層
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length)
    裡面放了(詞庫大小, embedding_dim, input_length=句子長度)

7.  Embedding
    實際上的運作超出本次範圍。簡單來說是給予一個16維的向量(例如本次設定的embedding_dim)
    讓每個單字都有這個向量，然後藉由labels給予的意義運算，讓類似的詞具有類似的向量，也就是這些詞都有類似的情緒。
    經過訓練後，就可以讓每個詞嵌入其中的情緒

    經過embeding()輸出是一個2D的數列，裡面是input_length與embedding_dim
    因此要經過Flatten()整理打成1D的輸入
    替代方案則是GlobalAveragePooling1D()，將向量平均化
    比起上面會只剩下16個輸入，相較之下或許速度更快，但準確率較低

8.  輸出用sigmoid，因為要判斷的是0正面意思，與1負面意思

9.  compile model，fit model
    fit包含資料，標籤，因此同樣的在驗證資料一樣要包含資料與標籤

    與之前imagegenerator較為不同，因為imagegenerator用資料夾已經分好labels了
    因此在fit_generator的時候不用特別帶入labels中的資料
    這邊的使用方法較類似Introduction_to_Tensorflow最一開始擬合直線範例的用法

'''

model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Flatten(),
        # tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
)

model.summary()
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)


# model.fit(
#     padded_s_tr_sentence, 
#     train_labels_ndarray,
#     epochs=num_epochs,
#     validation_data=(padded_s_te_sentence,test_labels_ndarray)
# )

'''
五、Extra內容

1.  看embedding層的細節
    先取出embedding的那一層，本案例中編號0，然後用.get_weight()[0]
    就可以看到實際上的權重。
    需要加上[0]，是因為若只有get_weight()
    會得到一個array([權重矩陣], dtype=xxx])，因此用[0]才提出第一項

2.  為了要繪出這個矩陣，我們需要從encode的數字反轉成單字
    在reverse_word_index_practice.py有更詳細的步驟說明
    
'''

e = model.layers[0]
weight = e.get_weights()[0]
print(weight)

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_text(text):
    temp = []
    for i in text:
        temp.append(reverse_word_index.get(i,'**'))
    decode = ' '.join(temp)
    return decode

decoded_texts = decode_text(padded_s_tr_sentence[0])
print(decoded_texts)
print(train_sentences[0])