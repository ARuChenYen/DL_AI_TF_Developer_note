import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from os import getcwd

file_dir = f'{getcwd()}\Sarcasm_Headlines_Dataset_v2.json'
full_json_list=[]
with open(file_dir, 'r') as f:
    for i in f.readlines():
        i = json.loads(i)
        full_json_list.append(i)

labels=[]
headline=[]
url=[]

for item in full_json_list:
    labels.append(item['is_sarcastic'])
    headline.append(item['headline'])
    url.append(item['article_link'])

tokenizer = Tokenizer(oov_token='<oov>')
tokenizer.fit_on_texts(headline)

headline_sequence = tokenizer.texts_to_sequences(headline)
padded_headline_sequence = pad_sequences(headline_sequence, padding='post')
print(padded_headline_sequence[0:2])
print(padded_headline_sequence.shape)
