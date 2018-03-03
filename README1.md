本文会通过 Keras 搭建一个深度卷积神经网络来识别一张图片是猫还是狗，在验证集上的准确率可以达到97.6%，建议使用显卡来运行该项目。本项目使用的 Keras 版本是2.0.4。如果你使用的是更高级的版本，可能会稍有参数变化。

# 猫狗大战

数据集来自 kaggle 上的一个竞赛：[Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)，训练集有25000张，猫狗各占一半。测试集12500张，没有标定是猫还是狗。


# 数据预处理

由于我们的数据集中含有异常图片，所以先需要将异常图片清除。
使用预训练resnet-50网络对训练集图片进行预测，得到所有图片的TOP40分类预测，并以csv文件格式保存
<pre><code>
model = ResNet50(weights='imagenet')
n=25000
for i in tqdm(range(n)):
    if i <(n/2):
        img_path = 'train/cat.%d.jpg' % i
    else:
        img_path ='train/dog.%d.jpg' % (i-(n/2))
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    Predicted_1 = decode_predictions(preds, top=40)[0][0][1]
    Predicted_2 = decode_predictions(preds, top=40)[0][1][1]
    Predicted_3 = decode_predictions(preds, top=40)[0][2][1]
    Predicted_4 = decode_predictions(preds, top=40)[0][3][1]
    Predicted_5 = decode_predictions(preds, top=40)[0][4][1]
    Predicted_6 = decode_predictions(preds, top=40)[0][5][1]
    Predicted_7 = decode_predictions(preds, top=40)[0][6][1]
    Predicted_8 = decode_predictions(preds, top=40)[0][7][1]
    Predicted_9 = decode_predictions(preds, top=40)[0][8][1]
    Predicted_10 = decode_predictions(preds, top=40)[0][9][1]
    Predicted_11 = decode_predictions(preds, top=40)[0][10][1]
    Predicted_12 = decode_predictions(preds, top=40)[0][11][1]
    Predicted_13 = decode_predictions(preds, top=40)[0][12][1]
    Predicted_14 = decode_predictions(preds, top=40)[0][13][1]
    Predicted_15 = decode_predictions(preds, top=40)[0][14][1]
    Predicted_16 = decode_predictions(preds, top=40)[0][15][1]
    Predicted_17 = decode_predictions(preds, top=40)[0][16][1]
    Predicted_18 = decode_predictions(preds, top=40)[0][17][1]
    Predicted_19 = decode_predictions(preds, top=40)[0][18][1]
    Predicted_20 = decode_predictions(preds, top=40)[0][19][1]
    Predicted_21 = decode_predictions(preds, top=40)[0][20][1]
    Predicted_22 = decode_predictions(preds, top=40)[0][21][1]
    Predicted_23 = decode_predictions(preds, top=40)[0][22][1]
    Predicted_24 = decode_predictions(preds, top=40)[0][23][1]
    Predicted_25 = decode_predictions(preds, top=40)[0][24][1]
    Predicted_26 = decode_predictions(preds, top=40)[0][25][1]
    Predicted_27 = decode_predictions(preds, top=40)[0][26][1]
    Predicted_28 = decode_predictions(preds, top=40)[0][27][1]
    Predicted_29 = decode_predictions(preds, top=40)[0][28][1]
    Predicted_30 = decode_predictions(preds, top=40)[0][29][1]
    Predicted_31 = decode_predictions(preds, top=40)[0][30][1]
    Predicted_32 = decode_predictions(preds, top=40)[0][31][1]
    Predicted_33 = decode_predictions(preds, top=40)[0][32][1]
    Predicted_34 = decode_predictions(preds, top=40)[0][33][1]
    Predicted_35 = decode_predictions(preds, top=40)[0][34][1]
    Predicted_36 = decode_predictions(preds, top=40)[0][35][1]
    Predicted_37 = decode_predictions(preds, top=40)[0][36][1]
    Predicted_38 = decode_predictions(preds, top=40)[0][37][1]
    Predicted_39 = decode_predictions(preds, top=40)[0][38][1]
    Predicted_40 = decode_predictions(preds, top=40)[0][39][1]
    
    Predicted_list_1.append(Predicted_1)
    Predicted_list_2.append(Predicted_2)
    Predicted_list_3.append(Predicted_3)
    Predicted_list_4.append(Predicted_4)
    Predicted_list_5.append(Predicted_5)
    Predicted_list_6.append(Predicted_6)
    Predicted_list_7.append(Predicted_7)
    Predicted_list_8.append(Predicted_8)
    Predicted_list_9.append(Predicted_9)
    Predicted_list_10.append(Predicted_10)
    Predicted_list_11.append(Predicted_11)
    Predicted_list_12.append(Predicted_12)
    Predicted_list_13.append(Predicted_13)
    Predicted_list_14.append(Predicted_14)
    Predicted_list_15.append(Predicted_15)
    Predicted_list_16.append(Predicted_16)
    Predicted_list_17.append(Predicted_17)
    Predicted_list_18.append(Predicted_18)
    Predicted_list_19.append(Predicted_19)
    Predicted_list_20.append(Predicted_20)
    Predicted_list_21.append(Predicted_21)
    Predicted_list_22.append(Predicted_22)
    Predicted_list_23.append(Predicted_23)
    Predicted_list_24.append(Predicted_24)
    Predicted_list_25.append(Predicted_25)
    Predicted_list_26.append(Predicted_26)
    Predicted_list_27.append(Predicted_27)
    Predicted_list_28.append(Predicted_28)
    Predicted_list_29.append(Predicted_29)
    Predicted_list_30.append(Predicted_30)
    Predicted_list_31.append(Predicted_31)
    Predicted_list_32.append(Predicted_32)
    Predicted_list_33.append(Predicted_33)
    Predicted_list_34.append(Predicted_34)
    Predicted_list_35.append(Predicted_35)
    Predicted_list_36.append(Predicted_36)
    Predicted_list_37.append(Predicted_37)
    Predicted_list_38.append(Predicted_38)
    Predicted_list_39.append(Predicted_39)
    Predicted_list_40.append(Predicted_40)
    
df = pd.DataFrame({'number':range(n),'pred1': Predicted_list_1,'pred2': Predicted_list_2,'pred3':Predicted_list_3,'pred4':Predicted_list_4,'pred5':Predicted_list_5,'pred6':Predicted_list_6,'pred7':Predicted_list_7,'pred8':Predicted_list_8,'pred9':Predicted_list_9,'pred10':Predicted_list_10,'pred11':Predicted_list_11,'pred12':Predicted_list_12,'pred13':Predicted_list_13,'pred14':Predicted_list_14,'pred15':Predicted_list_15,'pred16':Predicted_list_16,'pred17':Predicted_list_17,'pred18':Predicted_list_18,'pred19':Predicted_list_19,'pred20':Predicted_list_20,'pred21':Predicted_list_21,'pred22':Predicted_list_22,'pred23':Predicted_list_23,'pred24':Predicted_list_24,'pred25':Predicted_list_25,'pred26':Predicted_list_26,'pred27':Predicted_list_27,'pred28':Predicted_list_28,'pred29':Predicted_list_29,'pred30':Predicted_list_30,'pred31':Predicted_list_31,'pred32':Predicted_list_32,'pred33':Predicted_list_33,'pred34':Predicted_list_34,'pred35':Predicted_list_35,'pred36':Predicted_list_36,'pred37':Predicted_list_37,'pred38':Predicted_list_38,'pred39':Predicted_list_39,'pred40':Predicted_list_40})     
df.to_csv('leibie.csv', index=None)
df.head(10)
</code></pre>

再使用collections.Counter对出现的所有分类及其次数进行汇总统计，并将属于猫狗的分类加入一个集合
根据猫狗集合来判断哪些图片属于异常图片并可视化这些图片
<pre><code>
error_num =0
error_num_list = []
for i in tqdm(range(25000)):
    if data.loc[i].pred1 in dog_and_cat or data.loc[i].pred2 in dog_and_cat or data.loc[i].pred3 in dog_and_cat or data.loc[i].pred4 in dog_and_cat or data.loc[i].pred5 in dog_and_cat or data.loc[i].pred6 in dog_and_cat or data.loc[i].pred7 in dog_and_cat or data.loc[i].pred8 in dog_and_cat or data.loc[i].pred9 in dog_and_cat or data.loc[i].pred10 in dog_and_cat or data.loc[i].pred11 in dog_and_cat or data.loc[i].pred12 in dog_and_cat or data.loc[i].pred13 in dog_and_cat or data.loc[i].pred14 in dog_and_cat or data.loc[i].pred15 in dog_and_cat or data.loc[i].pred16 in dog_and_cat or data.loc[i].pred17 in dog_and_cat or data.loc[i].pred18 in dog_and_cat or data.loc[i].pred19 in dog_and_cat or data.loc[i].pred20 in dog_and_cat or data.loc[i].pred21 in dog_and_cat or data.loc[i].pred22 in dog_and_cat or data.loc[i].pred23 in dog_and_cat or data.loc[i].pred24 in dog_and_cat or data.loc[i].pred25 in dog_and_cat or data.loc[i].pred26 in dog_and_cat or data.loc[i].pred27 in dog_and_cat or data.loc[i].pred28 in dog_and_cat or data.loc[i].pred29 in dog_and_cat or data.loc[i].pred30 in dog_and_cat or data.loc[i].pred31 in dog_and_cat or data.loc[i].pred32 in dog_and_cat or data.loc[i].pred33 in dog_and_cat or data.loc[i].pred34 in dog_and_cat or data.loc[i].pred33 in dog_and_cat or data.loc[i].pred34 in dog_and_cat or data.loc[i].pred35 in dog_and_cat or data.loc[i].pred36 in dog_and_cat or data.loc[i].pred37 in dog_and_cat or data.loc[i].pred38 in dog_and_cat or data.loc[i].pred39 in dog_and_cat or data.loc[i].pred40 in dog_and_cat:
        pass
    else:
        error_num_list.append(data.loc[i].number)
        error_num += 1
print('error_num_list:',error_num_list)
</code></pre>

# 数据增强
使用keras的ImageDataGenerator来增加训练图片数量，防止过拟合。

<pre><code>
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
cats_num = 12478
dogs_num = 12479
num = 0
path = "train/"
dirs = os.listdir( path )
for file in tqdm(dirs):
    img = load_img(path+file) 
    x = img_to_array(img)  
    x = x.reshape((1,) + x.shape)  
    if file[:3] == 'cat':
        i = 0
        for batch in datagen.flow(x, batch_size=1,shuffle=False,
                                    save_to_dir='preview', save_prefix='cat'+ str(num), save_format='jpg'):
            i += 1
            if i > 5:
                break  # 否则生成器会退出循环
    else:
        i= 0
        for batch in datagen.flow(x, batch_size=1,shuffle=False,
                              save_to_dir='preview', save_prefix='dog'+ str(num), save_format='jpg'):
            i += 1
            if i > 5:
                break  # 否则生成器会退出循
    num += 1
</code></pre>

# 传入训练集和测试集并分割
<pre><code>
np.random.seed(2017)

n = 149709
X = np.zeros((n, 224, 224, 3), dtype=np.uint8)
y = np.zeros((n, 1), dtype=np.uint8)

for i in tqdm(range(n)):
    if i < catn:
        X[i] = cv2.resize(cv2.imread('preview/cat.%d.jpg' % i), (224, 224))
    else:
        X[i] = cv2.resize(cv2.imread('preview/dog.%d.jpg' % (i-catn)), (224, 224))


y[catn:] = 1

nn = 12500
Test = np.zeros((nn, 224, 224, 3), dtype=np.uint8)
for numT in tqdm(range(nn)):
    Test[numT] = cv2.resize(cv2.imread('test/%d.jpg' % (numT+1)), (224, 224))
    
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
</code></pre>

# 构建模型

载入预训练ResNet50模型，但不包括顶层分类器，冻结base_model所有层，并添加一个分类器

<pre><code>
base_model = ResNet50(input_tensor=Input((224, 224, 3)), weights='imagenet', include_top=False)


for layers in base_model.layers:
    layers.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.25)(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(base_model.input, x)

model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])
</code></pre>


# 训练模型

模型构件好了以后，我们就可以进行训练了，这里我们设置验证集大小为 20% ，也就是说训练集是20000张图，验证集是5000张图。

<pre><code>
result = model.fit(X_train, y_train, batch_size=32, epochs=8, validation_data=(X_valid, y_valid))
</code></pre>
>Train on 119767 samples, validate on 29942 samples
>Epoch 1/8
>119767/119767 [==============================] - 1583s - loss: 0.1430 - acc: 0.9423 - val_loss: 0.0708 - val_acc: 0.9733
>Epoch 2/8
>119767/119767 [==============================] - 1583s - loss: 0.1032 - acc: 0.9591 - val_loss: 0.0669 - val_acc: 0.9751
>Epoch 3/8
>119767/119767 [==============================] - 1578s - loss: 0.1031 - acc: 0.9602 - val_loss: 0.0657 - val_acc: 0.9752
>Epoch 4/8
>119767/119767 [==============================] - 1583s - loss: 0.0993 - acc: 0.9611 - val_loss: 0.0641 - val_acc: 0.9754
>Epoch 5/8
>119767/119767 [==============================] - 1586s - loss: 0.0996 - acc: 0.9610 - val_loss: 0.0622 - val_acc: 0.9762
>Epoch 6/8
>119767/119767 [==============================] - 1585s - loss: 0.0986 - acc: 0.9613 - val_loss: 0.0614 - val_acc: 0.9763
>Epoch 7/8
>119767/119767 [==============================] - 1585s - loss: 0.1000 - acc: 0.9608 - val_loss: 0.0623 - val_acc: 0.9761
>Epoch 8/8
>119767/119767 [==============================] - 1585s - loss: 0.0993 - acc: 0.9612 - val_loss: 0.0619 - val_acc: 0.9761

训练的过程大概要4个小时（在aws的P2.xrage实例上）在验证集上最高达到了97.6%的准确率。

# 预测测试集

模型训练好以后，我们就可以对测试集进行预测，然后提交到 kaggle 上看看最终成绩了。

<pre><code>
y_pred = model.predict(Test, verbose=1)
y_pred = y_pred.clip(min=0.005, max=0.995)

import pandas as pd
from keras.preprocessing.image import *

df = pd.read_csv("sample_submission.csv")

for i in tqdm(range(12500)):
    i = int(i)
    df.set_value(i, 'label', y_pred[i])

df.to_csv('pred.csv', index=None)
df.head(10)
</code></pre>

预测这里使用了clip将每个预测值限制到了 [0.005, 0.995] 个区间内，这个原因很简单，kaggle 官方的评估标准是 [LogLoss](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/details/evaluation)，对于预测正确的样本，0.995 和 1 相差无几，但是对于预测错误的样本，0 和 0.005 的差距非常大，是 15 和 2 的差别。参考 [LogLoss 如何处理无穷大问题](https://www.kaggle.com/wiki/LogLoss)，下面的表达式就是二分类问题的 LogLoss 定义。

$$\textrm{LogLoss} = - \frac{1}{n} \sum_{i=1}^n \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)\right]$$

![](https://raw.githubusercontent.com/ypwhs/resources/master/logloss.png)


# 总结

提交到 kaggle 以后，得分为0.5702，在全球排名中可以排到101/1314。我们如果要继续优化模型表现，可以对预训练模型进行微调（fine-tune），或者尝试其他模型。


