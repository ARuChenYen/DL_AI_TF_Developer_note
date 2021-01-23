import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd

def get_data(filename):
  # You will need to write code that will read the file passed
  # into this function. The first line contains the column headers
  # so you should ignore it
  # Each successive line contians 785 comma separated values between 0 and 255
  # The first value is the label
  # The rest are the pixel values for that picture
  # The function will return 2 np.array types. One with all the labels
  # One with all the images
  #
  # Tips: 
  # If you read a full line (as 'row') then row[0] has the label
  # and row[1:785] has the 784 pixel values
  # Take a look at np.array_split to turn the 784 pixels into 28x28
  # You are reading in strings, but need the values to be floats
  # Check out np.array().astype for a conversion
    with open(filename) as training_file:
        next(training_file)
        labels = []
        images= []
        for image in training_file:
            image = image.split(',')
            temp_label = image[0]
            temp_images = image[1:]
            
            t_labels = np.array(temp_label).astype(np.float32)
            t_images = np.array(temp_images).astype(np.float32)
            t_images = np.array_split(t_images,28)
            t_images = np.array(t_images)
            labels.append(t_labels)
            images.append(t_images)
        labels = np.array(labels)
        images = np.array(images)
    return images, labels

path_sign_mnist_train = f"{getcwd()}\sign_language_mnist\\test_train.csv"
path_sign_mnist_test = f"{getcwd()}\sign_language_mnist\\test_test.csv"
training_images, training_labels = get_data(path_sign_mnist_train)
testing_images, testing_labels = get_data(path_sign_mnist_test)

# Keep these
print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)

# Their output should be:
# (27455, 28, 28)
# (27455,)
# (7172, 28, 28)
# (7172,)

# In this section you will have to add another dimension to the data
# So, for example, if your array is (10000, 28, 28)
# You will need to make it (10000, 28, 28, 1)
# Hint: np.expand_dims

training_images = np.expand_dims(training_images,axis=3)
testing_images = np.expand_dims(testing_images,axis=3)

# Create an ImageDataGenerator and do Image Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)

train_generator = train_datagen.flow(
    training_images,
    y=training_labels,
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
)
    
# Keep These
print(training_images.shape)
print(testing_images.shape)
    
# Their output should be:
# (27455, 28, 28, 1)
# (7172, 28, 28, 1)

# Define the model
# Use no more than 2 Conv2D and 2 MaxPooling2D
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu',),
        tf.keras.layers.MaxPooling2D(2,2),
        ])

# Compile Model. 
model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['acc']
    )

# Train the Model
history = model.fit_generator(
    train_datagen,
    epochs= 2,
    validation_data=validation_datagen)

model.evaluate(testing_images, testing_labels, verbose=0)