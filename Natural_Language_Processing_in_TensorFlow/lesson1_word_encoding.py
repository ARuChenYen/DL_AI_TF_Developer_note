'''
一、文字處理
    文字比起圖片，要怎麼轉換成數字是個問題，完全不同意義的詞可能用完全相同的字母，ASCII編碼一樣
    因此要進行文字編碼

1.  Tensor flow提供編碼的api:Tokenizer
    Tokenizer幫助我們對文字加碼並且對於句子建立向量
    首先先實例化 
    tokenizer = Tokenizer(num_word = 100)
    num_word = 100 代表先建立100個編號。
    這個超參數設置也要小心，會影響訓練結果與時間

2.  將句子使用tokenizer編碼，用.fit_on_text()
    tokenizer.fit_on_text(sentences)

3.  用.word_index看編好的key與value數值

4.  標點符號並不會進入編碼，並且大小寫會看成同一個編碼
    

'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'I love my cat',
    'I love my dog',
    'You love my dog!',
    'Do you think my dog is amazing?',
]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)


'''

二、將文字轉換成句子
    接下來要進行句子轉換，才能進入訓練，而且要盡量保持句子長度一致

1.  用tokenizer.texts_to_sentences()轉換
    輸出會是一個list包含裡面的元素就是編碼後的文字
    但是要使用這個前，必須先經過.fit_on_tests()進行文字編碼
    不然輸出都會是空的list。

    因此在訓練時也要很注意要保持word index一致且包含所有文字
    不然句子編碼會錯誤，訓練會沒有意義。
    因此我們需要廣泛的字庫，避免句子出錯。
    對於還沒看過的詞，最好不要忽略，而是給予特殊的屬性

2.  新增oov_token="<OOV>"到Tokenizer中
    tokenizer = Tokenizer(num_word=100, oov_token="<OOV>")
    oov就是out of vocabulary，其實可以放任和自己想要的名稱，但不要與常用的單字一樣
'''

sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)

test_sentences = [
    'Sinra love my dog.',
    'she think my dog is really amazing.',
]

test_sentences = tokenizer.texts_to_sequences(test_sentences)
print(test_sentences)
# 如果沒有oov的話，這邊就會印出不完整的串列。

'''
三、Padding
    在進入神經網路訓練前，必須要讓數據一致化

1.  import pad_sequences
    在 tensorflow.kreas.preprocessing.sequence import pad_sequences

2.  實例化 padded = pad_sequences(sequences)
    記得這邊要給的是經過.texts_to_sequences，經過轉換後的串列
    直接給文字會出錯。
    印出結果會發現已經補0填空，讓每個句子都一樣長了

3.  其他參數 pad_sequences(sequences, padding='post', maxlen=5, truncating='post')
    padding，設定要從頭或者從尾填充句子，預設是從頭，用'post'改成從後
    maxlen，指定最長的句子，超過會截斷
    truncating，設定要從頭或者從尾截斷句子
'''

from tensorflow.keras.preprocessing.sequence import pad_sequences


padded = pad_sequences(sequences)
print(sequences)
print(padded)