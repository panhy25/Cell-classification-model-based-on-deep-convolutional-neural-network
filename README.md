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


