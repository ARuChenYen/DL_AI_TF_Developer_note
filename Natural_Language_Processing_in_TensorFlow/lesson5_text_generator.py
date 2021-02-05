'''

七、生成或者預測文本
    生成心的文本就類似於預測，神經網路學習到每個單詞與順序後，預測接下來會出現的單詞
    就是自動生成文本的過程
    
1.  過程都跟前面的類似。
    資料輸入 -> tokenizer -> sequences -> padded
    
2.  重要的概念，每個句子最後一個token當成labels的y，而前面的全部當成輸入的x
    也就是由輸入x得到結果y，因此要從前填充。
    因此我們還得把每個list進一步的拆成各種句子
    
    例如 [1, 26, 61, 60, 262, 13, 9, 10] 一串sequenced的句子
    還要進一步拆成 [1,26], [1,26,61], [1,26,61,60] ... 依此類推
    因為只要token數量大於2，都可以分成x input 跟y labels。
    所以在下面拆分句子的地方，會用for-loop去跑
    最終得到的結果類似 [[1,26],
                     [1,26,61],
                     [1,26,61,60],
                     ....]

3.  全部文本都經過上述方法拆分句子後，接下來就是要填充，把所有句子調整到長度一樣
    padded的時候，不但要找最長的句子，還要從前面來填充
    這樣才能把每個句子最後的token，也就是取作為label的部分方便取出
    
    xs = input_sequences[:,:-1]
    labels = input_sequences[:,-1]

4.  建立one-hot encode的layer
    ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
    這個指令會把labels轉換成0跟1的矩陣，只有在labels元素為編號的位址為1，其他地方都是零
    舉例來說 label=[1,2,0,5]
    經過轉換後就會變成 
    ys = [ [0,1,0,0,0,0],
           [0,0,1,0,0,0],
           [1,0,0,0,0,0],
           [0,0,0,0,0,5], ]


5.  在建立model或者之後預測時，都會出現input_length=max_sequence_len-1
    減1是因為最後一個元素會拿去當成labels，所以真正作為xs輸入的句子會少1
    model最後一層的輸出為 Dense(total_words, activation='softmax')
    使用的neuron會跟所有單字數量一樣多，所以當某個neuron被激發，就可以知道輸出的是哪個單字
    
    
6.  最後用model.predect_classes()，來預測文本
    預測後會得到一個y值，接著用for-loop去找整個tokenizer.word_index.items()的word, index
    如果預測值等於某個index，就知道這個單字會是預測的結果
    把這個單字加入原本句子(seed)中，然後再重複預測。看要幾次就可以知道原本文本後往下幾個單字會長怎樣

7.  預測越多會越來越不準確，組合的句子也會越來越詭異，甚至會變成都是同樣的單字重複出現
    增加語料庫會有所幫助。    

8.  三處可調整的超參數
    a.  Embedding(dim) embedding的維度
    b.  bidirectional(LSTM(cell_num)) lstm的cell數量，以及要不要雙向
        因為有些單字或者句子反過來並沒有太大意義。 例如 big dog, dog big
    c.  adam = Adam(lr=0.01) 調整adam的學習率
    都可以試試看對於訓練結果的影響

'''

import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np 

tokenizer = Tokenizer()

with open('Laurence_poetry.txt','r',encoding='utf8') as f:
    data = f.read()

# data="In the town of Athy one Jeremy Lanigan \n Battered away til he hadnt a pound. \nHis father died and made him a man again \n Left him a farm and ten acres of ground. \nHe gave a grand party for friends and relations \nWho didnt forget him when come to the wall, \nAnd if youll but listen Ill make your eyes glisten \nOf the rows and the ructions of Lanigans Ball. \nMyself to be sure got free invitation, \nFor all the nice girls and boys I might ask, \nAnd just in a minute both friends and relations \nWere dancing round merry as bees round a cask. \nJudy ODaly, that nice little milliner, \nShe tipped me a wink for to give her a call, \nAnd I soon arrived with Peggy McGilligan \nJust in time for Lanigans Ball. \nThere were lashings of punch and wine for the ladies, \nPotatoes and cakes; there was bacon and tea, \nThere were the Nolans, Dolans, OGradys \nCourting the girls and dancing away. \nSongs they went round as plenty as water, \nThe harp that once sounded in Taras old hall,\nSweet Nelly Gray and The Rat Catchers Daughter,\nAll singing together at Lanigans Ball. \nThey were doing all kinds of nonsensical polkas \nAll round the room in a whirligig. \nJulia and I, we banished their nonsense \nAnd tipped them the twist of a reel and a jig. \nAch mavrone, how the girls got all mad at me \nDanced til youd think the ceiling would fall. \nFor I spent three weeks at Brooks Academy \nLearning new steps for Lanigans Ball. \nThree long weeks I spent up in Dublin, \nThree long weeks to learn nothing at all,\n Three long weeks I spent up in Dublin, \nLearning new steps for Lanigans Ball. \nShe stepped out and I stepped in again, \nI stepped out and she stepped in again, \nShe stepped out and I stepped in again, \nLearning new steps for Lanigans Ball. \nBoys were all merry and the girls they were hearty \nAnd danced all around in couples and groups, \nTil an accident happened, young Terrance McCarthy \nPut his right leg through miss Finnertys hoops. \nPoor creature fainted and cried Meelia murther, \nCalled for her brothers and gathered them all. \nCarmody swore that hed go no further \nTil he had satisfaction at Lanigans Ball. \nIn the midst of the row miss Kerrigan fainted, \nHer cheeks at the same time as red as a rose. \nSome of the lads declared she was painted, \nShe took a small drop too much, I suppose. \nHer sweetheart, Ned Morgan, so powerful and able, \nWhen he saw his fair colleen stretched out by the wall, \nTore the left leg from under the table \nAnd smashed all the Chaneys at Lanigans Ball. \nBoys, oh boys, twas then there were runctions. \nMyself got a lick from big Phelim McHugh. \nI soon replied to his introduction \nAnd kicked up a terrible hullabaloo. \nOld Casey, the piper, was near being strangled. \nThey squeezed up his pipes, bellows, chanters and all. \nThe girls, in their ribbons, they got all entangled \nAnd that put an end to Lanigans Ball."
corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

print(tokenizer.word_index)
print(total_words)

# 拆分句子
input_sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
    # 這邊會取[0] 是因為用texts_to_sequences轉出來會是 list of list [[1, 26, 61, 60, 262, 13, 9, 10]]
    # 因此取第0個元素才能得到單純的list
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1] #配合i讓第一個取的會是token_list[:2] 也就是取0,1位元素。最少必須從2個開始
		input_sequences.append(n_gram_sequence)

# pad sequences 
# 先把所有的句子的長度建立一個list，然後從這個list中找出最大值，就會是最長的句子
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
xs, labels = input_sequences[:,:-1],input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

print(tokenizer.word_index['in'])
print(tokenizer.word_index['the'])
print(tokenizer.word_index['town'])
print(tokenizer.word_index['of'])
print(tokenizer.word_index['athy'])
print(tokenizer.word_index['one'])
print(tokenizer.word_index['jeremy'])
print(tokenizer.word_index['lanigan'])

model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_sequence_len-1)) # 減1是因為最後一個取作為label了，所以實際上進入網路的句子長度要少1
model.add(Bidirectional(LSTM(20)))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(xs, ys, epochs=500, verbose=1)


import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()

plot_graphs(history, 'accuracy')

seed_text = "Laurence went to dublin"
next_words = 100

# 這邊是用for-loop跑一百次，每次生成新詞後，重新加入句子看看下一次會生成什麼單字
# 也就是看看下一百個字會是什麼
# 因為Laurence是沒看過的外部詞彙，因此第一次的token_list只會有[134, 13, 59]三個

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    
    outputword = ''
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)