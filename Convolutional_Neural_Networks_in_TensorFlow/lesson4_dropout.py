'''
一、用dropout避免overfitting
    在訓練的時候，可以隨機捨棄掉一些neuron，來避免overfitting的情況
    因為鄰近的neuron可能常有類似的權重，將會導致overfit
    或者在上層進入本層時候over-weight了

1.  通常在輸入與輸出的那一層會讓dropout比較低(keep-prb保持比較多neuron)
    如果keep-prb=1.0 就是不dropout，通常在輸入與快要輸出的層保持keep-pob=1.0

2.  在keras實現方法為layer.Dropout(要丟棄的百分比)(x)
    x = keras.Dropout(0.2)(x) 就是丟棄了20%的神經元
'''

from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop

x = layers.Flatten()(last_output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_train_model.input, x)

model.compile(
    optimizer=RMSprop(lr=0.01),
    loss='binary_crossentropy',
    metrics=['acc'],
)