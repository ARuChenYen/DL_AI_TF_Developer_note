import tensorflow as tf

mnist = tf.keras.datasets.mnist

class myCallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, log={}):
        if (log.get('accuracy')>0.99):
            print('\nReached 99% accuracy so cancelling training!')
            self.model.stop_training = True

(x_train, y_train),(x_test, y_test) = mnist.load_data()

callbacks = myCallbacks()

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)
x_train = x_train/255
x_test = x_test/255

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()
model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
test_data = model.evaluate(x_test, y_test)

