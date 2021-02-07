'''
一、Time series
    什麼是time series? 隨時間變化的有序數列
    又可分為單變量與多變量

1.  常見使用機器學習在時變序列上的用途
    預測過去或未來，補值，異常值偵測，找到模式用於分析等等
    
2.  常用的時變數列特徵
    a.  趨勢
    b.  季節性
    c.  autucorrelation: 訊號在不同時間點自身相關
        在不同時間點會出現相似的複製訊號一樣
    d.  噪音

3.  時變分析不一定都這麼規律
    如果是stational的越多資料越好
    但non-stational的反而要好好選擇時間窗，越多資料反而會讓機器學習爆掉

4.  課程示範了幾種特徵的產生方法
    但是得詳細的理解一下code運作方式

'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


'''
5.  首先寫畫圖的函數
    a.  format決定圖示的表示與顏色，有固定設定格式可以參考文檔
    b.  label是決定圖中線條或者標點的名稱
    c.  legend設定就是看要不要呈現線條名稱，這個函數label是None，但如果之後有輸入label
        那麼就用legend標示出來，並且設定字體大小為14
    d.  start與end是把輸入的資料進行切片，預設是[0:]也就是全取的意思
    e.  xlabel與ylabel就是顯示x軸與y軸的名稱。還可以加上plt.title()就是圖的標題
    f.  grid(true)表示要把圖中的刻度表現出來
    h.  整張圖畫好後，會是一個物件，要用plt.show()才會顯示出來
'''
def plot_series(time, series, format='-', start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)


'''

6.  建立一個趨勢
    給一個時間點然後乘以斜率
    先用np.arange建立一個從零開始的序列，然後乘上斜率
'''

def trend(time, slope):
    return time*slope

time = np.arange(4*365+1) #方法類似range，建立一個從0到1460的序列
slope = 0.1
series = trend(time, slope)

plt.figure(figsize=(10,6)) #建立一張圖，大小為(10,6)
plot_series(time, series)
plt.show()


'''

8.  季節性的規律變化
    這邊要先去理解np.where的用法

    a.  np.where(condition[,x,y])
        直接來說就是如果condition為true，那麼就回傳x，否則就回傳y
        舉例來說，x = np.arange(10)
        condition = x > 5，結果是 array([False, False, False, False, False, False,  True,  True,  True, True])
        因此用np.where(x>5)就會得到 (array([6, 7, 8, 9], dtype=int64),)
    
    b.  在二維以上的情況，若只有設定條件，並不會直接回傳值，而是元素的下標
        例如 x = np.arange(10).reshape(2,-1)，reshape成2列，-1表示不限定大小
        所以x變成array([[0, 1, 2, 3, 4],
                       [5, 6, 7, 8, 9]])
        np.where(x>5)會得到 (array([1, 1, 1, 1], dtype=int64), 
                            array([1, 2, 3, 4], dtype=int64))
        顯示被取出的元素在 (1,1) (1,2) (1,3) (1,4) 四個地方，對應到就是6 7 8 9
    
    c.  np.where(condition,x,y)最多可以設定三個參數
        也就是(條件,符合的x處理,不符合的y處理)
        例如 np.where(x>5,x+2,x*10) 也就是條件為x>5，符合的x都加上2，不符合條件的y都*10
        所以輸出就變成了 array([[ 0, 10, 20, 30, 40],
                             [50,  8,  9, 10, 11]])

9.  seasonal_pattern():
    這邊先建立一個pattern會長怎樣，使用到上述的np.where
    設定條件為season_time <0.4，符合條件的就取 np.cos(season_time*2*np.pi)
    其實就是取 cos(season_time*2*pi)的意思，只是套用numpy的api
    如果不符合條件就取 1/np.exp(3*season_time)，取exp函數
    
10. seasonality():
    這邊就是設定時間上重複，phrase是相位差，period是週期，可以設定要多少一個週期
    ((time + phrase) % period) / period 就可以知道目前時間是整體週期的幾分之幾
    也確保不會大於1
    得到目前的時間點後，在使用函數seasonal_pattern就可以跑出現在的模式長怎樣

'''

def seasonal_pattern(season_time):
    return np.where(
        season_time <0.4,
        np.cos(season_time * 2 * np.pi),
        1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phrase=0):
    season_time = ((time + phrase) % period) / period
    return amplitude * seasonal_pattern(season_time)

time = np.arange(4*365+1)
amplitude = 40
phrase = 0
period = 365

series = seasonality(time, period, amplitude=amplitude, phrase=phrase)
plt.figure(figsize=(12,6))
plot_series(time, series)
plt.show()

    
'''
11. 季節性變化+趨勢
    就是把兩個函數加在一起就可以得到
'''

time = np.arange(4*365+1)
slope = 0.05
baseline= 10

series = baseline + trend(time, slope) + seasonality(time, period, amplitude=amplitude, phrase=phrase)
plt.figure(figsize=(12,6))
plot_series(time, series)
plt.show()

'''

12. 噪音
    這邊會使用到np.random，隨機產生亂數
    
    a.  x = np.random.RandomState(seed)
        x.randn(num)
        產生一個偽隨機的物件 x，然後用randn方法呼叫出一共num個平均0，標準差1的數
        RandomState產生的是用算法模擬出來的偽隨機，並且建立兩個RandomState卻使用同樣的seed
        那麼呼叫出來的隨機數將會一樣。
        seed是用來初始化這個物件，預設是None
        不用rand是因為rand只會產生0-1，不會有負的
    
    b.  下面的範例就是先建立一個隨機數的物件然後去呼叫，隨機數數量就是time時間的個數，因此用len(time)
        noise_level是決定噪音的震幅
        
    c.  讓訊號加上噪音也很簡單，用加的就可以了
'''

def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

noise_level = 5
seed = 42
noise = white_noise(time, noise_level=noise_level, seed=seed)

plt.figure(figsize=(12,6))
plot_series(time, noise)
plt.show()

series += noise
plt.figure(figsize=(12,6))
plot_series(time, series)
plt.show()