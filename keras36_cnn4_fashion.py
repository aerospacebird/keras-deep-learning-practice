
from tensorflow.keras.datasets import fashion_mnist


# 실습 만들기
import pandas as pd
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense,Dropout, Flatten
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense
)
import time
from sklearn.metrics import accuracy_score


#1. 데이터
model = Sequential()

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)  # (10000, 28, 28) (10000,)
#exit()
print(np.max(x_train), np.min(x_train)) #255 0 #제일 밝은 값과 제일 어두운 값
print(np.max(x_test), np.min(x_test))   #255 0  #0=검정 255=흰색

##### 스케일링 1 >>>>>>>>>>>0~255 → 0~1 변경 ###########
x_train = x_train/255.  #점찍으면 플로트(실수)표현
x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) #1.0 0.0 #0→0.0=⚫검정
print(np.max(x_test), np.min(x_test))   #1.0 0.0 #255→1.0=⚪흰색

#exit()


# ####스케일링 2 >>>>>>>>>>>>>0~255 → -1~1 변경 #############
# x_train = (x_train-127.5)/127.5
# x_test = (x_test-127.5)/127.5
# print(np.max(x_train), np.min(x_train)) #1.0 -1.0
# print(np.max(x_test), np.min(x_test))   #1.0 -1.0

x_train = x_train.reshape(-1, 28,28,1)
x_test = x_test.reshape(-1, 28,28,1)
print(x_train.shape, x_test.shape)  #(60000, 28, 28, 1) (10000, 28, 28, 1)

# exit()

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)

y_train = y_train.reshape(60000,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

print(y_train.shape, y_test.shape)

#exit()

# #2. 모델구성
# model = Sequential()
# model.add(Conv2D(64,(3,3), input_shape=(28,28,1)))  #26,26,64
# model.add(Conv2D(filters=32, kernel_size=(3,3),activation='relu'))  #(24,24,32)#filters=32  channels
# model.add(Dropout(0.2))
# model.add(Conv2D(32,(2,2,), activation='relu')) #(23,23,32)
# model.add(Conv2D(16,(2,2,), activation='relu')) #(22,22,16)
# model.add(Dropout(0.2))
# model.add(Conv2D(32,(2,2,), activation='relu')) #(21,21,32)
# model.add(Dropout(0.2))
# model.add(Conv2D(16,(2,2,), activation='relu')) #(20,20,16) #6400
# model.add(Flatten())                        #납작하게 펴기📦➡️📏

# model.add(Dense(units=32,activation='relu')) #👉 뉴런 32개를 만들겠다
# model.add(Dropout(0.2))
# model.add(Dense(units=16, input_shape =(32,),activation='relu'))
# model.add(Dense(10,activation='softmax'))   #(10,)

# model.summary()


# Model: "sequential_1"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 26, 26, 64)        640       
                                                                 
#  conv2d_1 (Conv2D)           (None, 24, 24, 32)        18464     
                                                                 
#  dropout (Dropout)           (None, 24, 24, 32)        0         
                                                                 
#  conv2d_2 (Conv2D)           (None, 23, 23, 32)        4128      
                                                                 
#  conv2d_3 (Conv2D)           (None, 22, 22, 16)        2064      
                                                                 
#  dropout_1 (Dropout)         (None, 22, 22, 16)        0         
                                                                 
#  conv2d_4 (Conv2D)           (None, 21, 21, 32)        2080      
                                                                 
#  dropout_2 (Dropout)         (None, 21, 21, 32)        0         
                                                                 
#  conv2d_5 (Conv2D)           (None, 20, 20, 16)        2064      
                                                                 
#  flatten (Flatten)           (None, 6400)              0         
                                                                 
#  dense (Dense)               (None, 32)                204832    
                                                                 
#  dropout_3 (Dropout)         (None, 32)                0         
                                                                 
#  dense_1 (Dense)             (None, 16)                528       
                                                                 
#  dense_2 (Dense)             (None, 10)                170       
                                                                 
# =================================================================
# Total params: 234,970
# Trainable params: 234,970
# Non-trainable params: 0
# #2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense
)

model = Sequential()

# ============================================================
# Block 1
# ============================================================
model.add(Conv2D(
    32, (3,3),
    padding='same',
    activation='relu',
    input_shape=(28,28,1)
))

model.add(BatchNormalization())

model.add(Conv2D(
    32, (3,3),
    padding='same',
    activation='relu'
))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


# ============================================================
# Block 2
# ============================================================
model.add(Conv2D(
    64, (3,3),
    padding='same',
    activation='relu'
))

model.add(BatchNormalization())

model.add(Conv2D(
    64, (3,3),
    padding='same',
    activation='relu'
))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


# ============================================================
# Classifier
# ============================================================
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(10, activation='softmax'))

model.summary()


# Model: "sequential_1"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 28, 28, 32)        320       
                                                                 
#  batch_normalization (BatchN  (None, 28, 28, 32)       128       
#  ormalization)                                                   
                                                                 
#  conv2d_1 (Conv2D)           (None, 28, 28, 32)        9248      
                                                                 
#  batch_normalization_1 (Batc  (None, 28, 28, 32)       128       
#  hNormalization)                                                 
                                                                 
#  max_pooling2d (MaxPooling2D  (None, 14, 14, 32)       0         
#  )                                                               
                                                                 
#  dropout (Dropout)           (None, 14, 14, 32)        0         
                                                                 
#  conv2d_2 (Conv2D)           (None, 14, 14, 64)        18496     
                                                                 
#  batch_normalization_2 (Batc  (None, 14, 14, 64)       256       
#  hNormalization)                                                 
                                                                 
#  conv2d_3 (Conv2D)           (None, 14, 14, 64)        36928     
                                                                 
#  batch_normalization_3 (Batc  (None, 14, 14, 64)       256       
#  hNormalization)                                                 
                                                                 
#  max_pooling2d_1 (MaxPooling  (None, 7, 7, 64)         0         
#  2D)                                                             
                                                                 
#  dropout_1 (Dropout)         (None, 7, 7, 64)          0         
                                                                 
#  flatten (Flatten)           (None, 3136)              0         
                                                                 
#  dense (Dense)               (None, 128)               401536    
                                                                 
#  batch_normalization_4 (Batc  (None, 128)              512       
#  hNormalization)                                                 
                                                                 
#  dropout_2 (Dropout)         (None, 128)               0         
                                                                 
#  dense_1 (Dense)             (None, 64)                8256      
                                                                 
#  dropout_3 (Dropout)         (None, 64)                0         
                                                                 
#  dense_2 (Dense)             (None, 10)                650       
                                                                 
# =================================================================
# Total params: 476,714
# Trainable params: 476,074
# Non-trainable params: 640
# _________________________________________________________________

#3. 컴파일 , 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam',
            metrics = ['acc'],
            )   

start_time = time.time()
model.fit(x_train,y_train, epochs=50, batch_size=128,
        verbose = 1, 
        validation_split = 0.2,
          )
end_time = time.time()


#4.평가, 예측
print("===============================model.evaluate================================")
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss:', loss[0] )
print('acc:', loss[1])

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict,axis=1,)#.reshape(-1,1)
y_test = np.argmax(y_test,axis=1,)#.reshape(-1,1)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('걸린시간 : ', round(end_time-start_time,2), '초')

###########################################################################################
# loss: 0.2516067326068878
# acc: 0.9350000023841858

# accuracy_score :  0.935
# 걸린시간 :  108.32 초

########################################################################################
# loss: 0.2687610983848572
# acc: 0.9311000108718872
# 313/313 [==============================] - 0s 1ms/step
# accuracy_score :  0.9311
# 걸린시간 :  157.86 초