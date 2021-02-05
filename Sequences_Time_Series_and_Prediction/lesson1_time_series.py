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

'''