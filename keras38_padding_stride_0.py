
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

#2. 모델구성
model = Sequential()
model.add(Conv2D(10,(2,2),input_shape =(10,10,1),   #9, 9, 10
                 strides=1,                         #5, 5, 10
                padding='same',                     #너의빈공간을 0으로 채운다 / 10, 10, 10

))
model.add(MaxPooling2D())


model.add(Conv2D(filters=9, kernel_size=(3,3),      #7, 7, 9
                 strides=2,                         #
                padding='valid',                    #디폴트 /8, 8, 9
))

model.summary()