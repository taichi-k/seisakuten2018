# emoji recognition by CNN

絵文字７種類＋裏面('null')＋絵文字なし('none')の９ラベルで学習．

kerasモデルで作成．構成は以下の通り．

```
model = Sequential()
model.add(Conv2D(12,kernel_size=(3,3),activation='relu',input_shape=(64,64,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(18,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(24,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(48,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32,activation='relu'))
model.add(Dense(emo_n,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
return model
```

keras環境によっては _load_model_ がうまくいかないかも．
その場合はモデルのパラメータだけを保存してアップロードしなおすので，上記のモデルを作成しパラメータをロードさせる．

使用サンプルとして _predictor.py_　を用意した．
predict部分は # で囲った部分のみ．

モデルはデータをシャッフルして学習させたもの２つ用意した．
1つだと少し安定しないため，2つの出力の平均を取ることにする．
また，出力はsoftmaxでラベルごとの確率として出るが，これが0.8以下であるときは絵文字の予測をせず'none'とする．
