import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


sentences = [
    'I love my cat',
    'I love my dog',
    'You love my dog!',
    'Do you think my dog is amazing?',
]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

test_sentences = [
    'Sinra love my dog.',
    'she think my dog is really amazing.',
]

test_sentences = tokenizer.texts_to_sequences(test_sentences)

padded = pad_sequences(sequences, padding='post')
print(sequences)
print(padded)

print(word_index)
# 結果會是把整個字典打出來 
# {'<OOV>': 1, 'my': 2, 'love': 3, 'dog': 4, 'i': 5, 'you': 6, 'cat': 7, 'do': 8, 'think': 9, 'is': 10, 'amazing': 11}

for j in word_index:
    print(j)
# 這個只會把字典中的key打出來，但是value不會出現
# 等效 for j in word_index.key():

for i in word_index.items():
    print(i)
# 這個會把每組以('key':value)的型態輸出
# ('<OOV>', 1) ...

for k in word_index.values():
    print(k)
# 這個就會把每個value打印出來，但是key不會


# reverse method 1
temp=[]
for item in word_index.items():
    temp.append((item[1], item[0]))
reverse_word_index = dict(temp)

#reverse method 2
reverse_word_index2 = dict([(value, key) for (key, value) in word_index.items()])
print(reverse_word_index)
print(reverse_word_index2)
# method1 等效 method2，兩者輸出一樣
# {1: '<OOV>', 2: 'my', 3: 'love', 4: 'dog', 5: 'i', 6: 'you', 7: 'cat', 8: 'do', 9: 'think', 10: 'is', 11: 'amazing'}

def decode_text(text):
    temp = []
    for i in text:
        temp.append(reverse_word_index.get(i,'**'))
        # .get(i, '**')，會回去reverse_word_index找，如果有找到就回傳value，沒有就回傳**
        # 先用個list把反轉的詞存起來，最後用.join合併成str
    decode = ' '.join(temp)
    return decode

decoded_texts = decode_text(padded[0])
print(decoded_texts)