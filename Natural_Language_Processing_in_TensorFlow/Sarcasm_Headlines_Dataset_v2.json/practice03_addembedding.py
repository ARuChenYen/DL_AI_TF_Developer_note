import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy as np
from os import getcwd

vocab_size = 10000
embedding_dim = 16
max_length = 32
trunc_type = 'post'
padding_type = 'post'
oov_token = '<oov>'
num_epochs = 30
training_size = 20000




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


# Set training data and testing data
training_articles = headline[0:training_size]
training_labels = labels[0:training_size]
testing_articles = headline[training_size:]
testing_labels = labels[training_size:]
training_labels_ndarray = np.array(training_labels)
testing_labels_ndarray = np.array(testing_labels)

# Training and testing data tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<oov>')
tokenizer.fit_on_texts(training_articles)
word_index = tokenizer.word_index

training_sequence = tokenizer.texts_to_sequences(training_articles)
training_padding = pad_sequences(
    training_sequence, padding=padding_type, maxlen=max_length, truncating=trunc_type,
    )
print(training_padding[0:2])

testing_sequence = tokenizer.texts_to_sequences(testing_articles)
testing_padding = pad_sequences(
    testing_sequence, padding=padding_type, maxlen=max_length, truncating=trunc_type,
    )

# Create DNN model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length,),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ]
)

model.summary()

# Compile and fit model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)

history = model.fit(
    training_padding, 
    training_labels_ndarray, 
    epochs=num_epochs, 
    validation_data=(testing_padding, testing_labels_ndarray),
    verbose=2,
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