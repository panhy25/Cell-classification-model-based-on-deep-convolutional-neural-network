# Cell-classification-model-based-on-deep-convolutional-neural-network
人生中的第一个CNN建立和第一个web app  
这是一份写给自己的教程，也算是对工作的总结。当然，这也是给她的情书。  
在本项目中我们制作了一个CNN网络，在kaggle来进行学习计算，最后得出一个还不错的准确率，并发布为web app。这个项目不复杂，但给了我们很多收获。本文将分为以下几个部分来叙述：kaggle、图像增强、CNN、CNN的参数、streamlit、阿里云、各种各样有关深度卷积的图这些方面来叙述。
## 一 kaggle
### 什么是kaggle
kaggle是一个可以托管数据库，编写运行和分享代码的平台。最重要的优势包括免费的数据库和GPU，不用翻墙就能用的畅快体验，各位大佬已经公开的相关代码。不过除了这些，最吸引初学者的可能是不需要库的安装和环境变量的调试，这真的太让人开心了。  
### 如何使用kaggle
kaggle需要你的邮箱和手机号码，只要提供手机号码，kaggle将开放GPU和TPU供你使用。使用时长为每周30小时，可能不是很够，建议找个女朋友要她的手机号。  
### 一些小提示
kaggle最好用压缩包上传你的本地数据集，不然会慢的要死  
运行代码的时候要看着它，别代码都停了也不知道。当然更推荐save version，这个一了百了解决问题
用数据库之前多看看前辈的代码，会有意想不到的收获。比如听到大伙怒斥test集和train集不是一个难度，就应该放弃这个数据集了    
要使用kaggle要注意文件的格式是ipynb，不能import自己的库，所以那种pycharm里的大工程不是很合适
## 二 图像增强
```
validation_split=0.15,
rescale=1. / 255,  # 重缩放因子
shear_range=0.2,  # 剪切强度（以弧度逆时针方向剪切角度）
zoom_range=0.2,  # 随机缩放范围
vertical_flip=True,  #随机上下翻转
horizontal_flip=True  # 随机水平翻转
```
遗憾的是我们不懂白细胞，只能每一次都试试。但是有一个原则，就是不要不符合实际的去做。下面是图片增强的效果：
![image](https://github.com/panhy25/Cell-classification-model-based-on-deep-convolutional-neural-network/blob/main/blood-cell-image-git/change.png)
## 三 CNN
CNN结构如下：  
![image](https://github.com/panhy25/Cell-classification-model-based-on-deep-convolutional-neural-network/blob/main/blood-cell-image-git/cnn.png)  
其实说实话，我也不是很懂卷积层都在干什么，但是一次次的实验证明了这个CNN还可以。下面分批展示代码和注释：  
```
# 看看kaggle的GPU有没有跑
import tensorflow as tf
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
print(tf.__version__)
a = tf.constant(1.)
b = tf.constant(2.)
print(a+b)
print('GPU:', tf.test.is_gpu_available())
```
```
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import glob, os, random
```
```
# 统一定义图像像素的宽度和高度
img_width, img_height = 224, 224

# 定义训练集、验证集的图形路径（文件夹路径即可）
train_data_dir = '../input/blood-cells/dataset2-master/dataset2-master/images/TRAIN'
test_data_dir = '../input/blood-cells/dataset2-master/dataset2-master/images/TEST'

# 模型训练的参数设置
epochs = 50  # 迭代次数
batch_size = 32  # 每个批量观测数

# 图像输入维度设置
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
```
```
train_datagen = ImageDataGenerator(validation_split=0.15,
                                   rescale=1. / 255,  # 重缩放因子
                                   shear_range=0.2,  # 剪切强度（以弧度逆时针方向剪切角度）
                                   zoom_range=0.2,  # 随机缩放范围
                                   vertical_flip=True,  #随机上下翻转
                                   horizontal_flip=True  # 随机水平翻转
                                  )
test_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    subset="training",
                                                    target_size=(img_width, img_height), 
                                                    batch_size=batch_size,
                                                    class_mode='categorical',  # 指定分类模式
                                                    classes=['EOSINOPHIL','LYMPHOCYTE','MONOCYTE','NEUTROPHIL']
                                                   )
val_generator = train_datagen.flow_from_directory(train_data_dir,
                                                  subset="validation",
                                                  target_size=(img_width, img_height),
                                                  batch_size=batch_size,
                                                  class_mode='categorical',  # 指定分类模式
                                                  classes=['EOSINOPHIL','LYMPHOCYTE','MONOCYTE','NEUTROPHIL']
                                                  )
test_generator = test_datagen.flow_from_directory(test_data_dir,
                                                  target_size=(img_width, img_height),
                                                  class_mode='categorical',  # 指定分类模式
                                                  classes=['EOSINOPHIL','LYMPHOCYTE','MONOCYTE','NEUTROPHIL']
                                                  ) 
```
```
# CNN的构建
model = Sequential()

# 添加第一个卷积层/最大池化层
model.add(Conv2D(filters=64,
          kernel_size=(3, 3),
          input_shape=input_shape, 
          activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 

# 添加第二个卷积层/最大池化层
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 添加第三个卷积层/最大池化层
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 由于卷积层是 2D 空间，训练时需要将数据展平为 1D 空间
model.add(Flatten())  # 添加展平层
model.add(Dense(units=128, activation='relu'))  # 添加全连接层128个神经元
model.add(Dropout(0.5))  # 添加丢弃层，防止过拟合

# 输出层：最后一层，神经元控制输出的维度，并指定分类激活函数
model.add(Dense(units=4, activation='softmax'))  # 指定分类激活函数

model.summary()
```
```
model.compile(loss='categorical_crossentropy',  # 指定损失函数类型
              optimizer='rmsprop',  # 优化器
              metrics=['accuracy'])  # 评价指标
```
```
history = model.fit(train_generator,
                    epochs=epochs,
                    validation_data=val_generator,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            patience=10,
                            restore_best_weights=True
                            )
                        ]
                    )
model.save('model.h5')
```
以上，CNN的构建已经结束，我们可以跑跑代码看看效果了。跑完了用下面的代码可以展示acc和loss的变化，值得一试。
```
import matplotlib.pyplot as plt
%matplotlib inline
training_loss = history.history['loss']
test_loss = history.history['val_loss']
# 创建迭代数量
epoch_count = range(1, len(training_loss) + 1)
# 可视化损失历史
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
```
```
train_acc = history.history['accuracy']
test_acc = history.history['val_accuracy']
epoch_counts = range(1, len(train_acc)+1)
plt.plot(epoch_counts, train_acc, 'r--', marker='^')
plt.plot(epoch_counts, test_acc, linestyle='-', marker='o', color='y')
plt.title('accuracy condition')
plt.legend(['train_acc', 'test_acc'])
plt.xlabel('epochs')
plt.ylabel('acc')
```
以上代码均已上传，可以直接查看。
## CNN的参数
这个是本项目的重中之重，或者说是最精华的地方也不过分，毕竟训练了一个不准的模型除了为你自己积累经验似乎没什么别的作用。那么到底要怎么调参，就是接下来讨论的地方。
```
optimizer='rmsprop'
model.add(Dense(units=4, activation='softmax'))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
validation_split=0.15
model.add(Dropout(0.5))
```
以上是我觉得有必要折腾的地方，最重要的就是optimizer，这个换一换会有很大影响，可以在网上找到好多关于这些函数的解析，但是挨个试试总没错。另外有几个卷积层，几个卷积核，验证集的划分，dropout的值都可以换着试试。  
epoch直接1000，跑到哪算哪，不跑怎么知道哪能收敛。
## streamlit
这真的是拯救了这个项目的好网站，建议给开发者磕一个。  
### streamlit是什么
Streamlit 是一个用于机器学习、数据可视化的 Python 框架，它能几行代码就构建出一个精美的在线 app 应用。非常适合不懂构建网站的小白使用。
### streamlit代码
```
import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from keras.utils import image_utils
from keras.applications import *
import cv2
import streamlit as st
import os
import tensorflow as tf
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def decode_predictions(file_path):
    image = image_utils.load_img(file_path, target_size=(224,224))
    x = image_utils.img_to_array(image)
    x = np.expand_dims(x, axis=0)
    x = x/255

    model = tf.keras.models.load_model('/root/my_st/model.h5')
    y = model(x)
    y = np.argmax(y, axis=1)
    return y

st.title('白细胞分类器')
st.header('白细胞图像上传')

uploaded_file = st.file_uploader("上传一张图片", type=['jpg','png','jpeg'])
st.sidebar.expander('')
st.sidebar.subheader('kaggle账号：@liyf279 @panhy25')
st.sidebar.subheader('github账号：@panhy25')
st.sidebar.subheader('由于版本的不同，notebook中acc的值和下面展示的结果不同，但应该没有过大的差异')
st.sidebar.write('https://www.kaggle.com/code/liyf279/blood-cell-predict')
st.sidebar.write('https://www.kaggle.com/liyf279/streamlit')
st.sidebar.write('Train_acc:92.09%')
st.sidebar.write('Val_acc:92.16%')
st.sidebar.write('Tset_acc:93.12%')
st.sidebar.subheader('Our CNN model')
imag1 = Image.open('/root/one.jpg')
st.sidebar.image(imag1, caption=None, use_column_width=True)
imag2 = Image.open('/root/two.png')
st.sidebar.image(imag2, caption=None, use_column_width=True)

if uploaded_file is not None:
    st.write('上传成功')
    # 将传入的文件转为Opencv格式
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    if st.button('test'):
        # 展示图片
        st.image(opencv_image, channels="BGR")
        # 保存图片
        cv2.imwrite('test.jpg', opencv_image)
        file_path = 'test.jpg'
        q = decode_predictions(file_path)
        dic = {0: 'EOSINOPHIL', 1: 'LYMPHOCYTE', 2: 'MONOCYTES', 3: 'NEUTROPHIL'}
        for keys in dic:
            if keys == q:
                st.write(dic[keys])
                st.text('Well Done!')
                st.balloons()
```
## 阿里云
阿里云的部署不是一件容易的事，详细学习的话有很多内容，这里如果只想简单有一个网站，推荐
https://jackiexiao.github.io/blog/%E6%8A%80%E6%9C%AF/%E4%BD%BF%E7%94%A8docker%E5%92%8Cstreamlit%E9%98%BF%E9%87%8C%E4%BA%91%E6%9C%8D%E5%8A%A1%E5%99%A8%E9%83%A8%E7%BD%B2%E7%AE%80%E5%8D%95%E7%9A%84%E6%BC%94%E7%A4%BA%E7%BD%91%E9%A1%B5/



