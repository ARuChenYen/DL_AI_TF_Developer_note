'''
一、Multi-classification
    不只有2次元分類，而是多個類別要分類
1.  使用Rock-Paper-Scissors
    這是一個用CGI產生各種手的圖庫，有各種顏色性別種族跟年齡等。有2,892張圖，每張圖都是300*300，24-BIT
    但總共有剪刀石頭布三種(三類)

2.  改變地方
    A.  ImageGenerator中class_mode改成'categorical'
    B.  輸出層從sigmoid改成softmax
    C.  compile的loss改成'categorical_crossentropy'

'''

