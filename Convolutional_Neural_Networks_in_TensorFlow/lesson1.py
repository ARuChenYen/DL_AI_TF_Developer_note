'''
一、訓練完後查看訓練歷史
    訓練完model後可以查看訓練的loss跟accuracy
1.  history = model.fit(...)
2.  acc = history.history['acc']
3.  val_acc = history.history['val_acc']
4.  epochs = range(len(acc)) 取得epoch的數量，用range是因為要產生序列畫圖
5.  plt.plot(epoch,acc) 畫圖
6.  plt.figure()

二、檔案操作
1.  getcwd() 會獲得當前目錄的位址，本處是D:\code\coursera\DLAI_TF_Developer\Convolutional_Neural_Networks_in_ensorFlow


'''
import os
import zipfile
import random
# import tensorflow as tf
import shutil
# from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd

path_cats_and_dogs_zip = f"{getcwd()}\cats_and_dogs.zip"
path_cats_and_dogs =  f"{getcwd()}\cats_and_dogs"
# getcwd() 會獲得當前目錄的位址，本處是D:\code\coursera\DLAI_TF_Developer\Convolutional_Neural_Networks_in_ensorFlow
temp_dir = r'\tmp'
path_test = f"{getcwd()}{temp_dir}"
# shutil.rmtree(path_test) #刪掉路徑中的目錄。

# local_zip = path_cats_and_dogs_zip
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall()
# zip_ref.close()

print(len(os.listdir(f"{path_cats_and_dogs}\cats")))
print(len(os.listdir(f"{path_cats_and_dogs}\dogs")))

path_training = f"{path_cats_and_dogs}" + r'\training'
path_testing = f"{path_cats_and_dogs}" + r'\testing'

try:
    os.mkdir(f"{path_training}")
    os.mkdir(f"{path_training}\cats")
    os.mkdir(f"{path_training}\dogs")
    os.mkdir(f"{path_testing}")
    os.mkdir(f"{path_testing}\cats")
    os.mkdir(f"{path_testing}\dogs")
    os.mkdir(f"{path_cats_and_dogs}\zero_size_files")
except OSError as identifier:
    pass

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    source_files_list = os.listdir(SOURCE)
    source_num = int(len(source_files_list)*SPLIT_SIZE)
    tr_sample = random.sample(source_files_list, source_num)

    for i in source_files_list:
        filepath = f"{SOURCE}\\" + i
        if os.path.getsize(filepath) != 0 and i in tr_sample:
            copyfile(filepath, TRAINING)
        elif os.path.getsize(filepath) != 0 and i not in tr_sample:
            copyfile(filepath, TESTING)

CAT_SOURCE_DIR = f"{path_cats_and_dogs}\cats"
TRAINING_CATS_DIR = f"{path_training}\cats"
TESTING_CATS_DIR = f"{path_testing}\cats"
DOG_SOURCE_DIR = f"{path_cats_and_dogs}\dogs"
TRAINING_DOGS_DIR = f"{path_training}\dogs"
TESTING_DOGS_DIR = f"{path_testing}\dogs"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)