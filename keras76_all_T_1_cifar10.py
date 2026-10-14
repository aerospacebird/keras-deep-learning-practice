import warnings
warnings.filterwarnings("ignore")

from keras.applications import VGG16, VGG19
from keras.applications import ResNet50, ResNet50V2 
from keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from keras.applications import DenseNet201, DenseNet121, DenseNet169
from keras.applications import InceptionResNetV2, InceptionV3
from keras.applications import MobileNet, MobileNetV2
from keras.applications import MobileNetV3Small, MobileNetV3Large
from keras.applications import NASNetLarge, NASNetMobile
from keras.applications import EfficientNetB0,EfficientNetB1,EfficientNetB2
from keras.applications import Xception

model_list = [
        VGG19(include_top=False, input_shape=(32,32,3)),
        ResNet50(include_top=False, input_shape=(32,32,3)),
        ResNet101(include_top=False, input_shape=(32,32,3)),
        DenseNet201(include_top=False, input_shape=(32,32,3)),
        # InceptionV3(),InceptionResNetV2(),
        MobileNetV2(include_top=False, input_shape=(32,32,3)),
        EfficientNetB0(include_top=False, input_shape=(32,32,3))
]

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf
import time
from sklearn.metrics import accuracy_score
import random

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

from tensorflow.keras.datasets import cifar10, cifar100

for model_i in model_list:
    model_i.trainable = False

    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(10,activation='softmax'))

    (x_train,y_train), (x_test, y_test) = cifar10.load_data()

    #scaling
    x_train = x_train/255.
    x_test = x_test/255.

    #3. compile,train
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['acc'])

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    es = EarlyStopping()


    model.fit(x_train, y_train, epochs=1000, batch_size=32,
              verbose=0,
              validation_split=0.2,
              callbacks=[es])


    #4. test,predict
    loss = model.evaluate(x_test, y_test, verbose=0)
    print("=================================")
    print("모델명 : ", model_i.name, 'loss : ', loss[0], 'acc :', loss[1])

################################################################################################################################
# =================================
# 모델명 :  vgg19 loss :  1.3092858791351318 acc : 0.5418999791145325
# =================================
# 모델명 :  resnet50 loss :  1.828803300857544 acc : 0.3377000093460083
# =================================
# 모델명 :  resnet101 loss :  1.9906333684921265 acc : 0.2770000100135803
# =================================
# 모델명 :  densenet201 loss :  1.168580412864685 acc : 0.5996999740600586
# =================================
# 모델명 :  mobilenetv2_1.00_224 loss :  1.9077343940734863 acc : 0.31060001254081726
# =================================
# 모델명 :  efficientnetb0 loss :  2.3109893798828125 acc : 0.10000000149011612
###############################################################################################################################