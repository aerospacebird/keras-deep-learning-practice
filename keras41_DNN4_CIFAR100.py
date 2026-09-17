#from keras.datasets import fashion_mnist, cifar100

from keras.datasets import cifar100

# 실습 만들기
import pandas as pd
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
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

# from tensorflow.keras.datasets import cifar10

# # CIFAR-10 데이터 불러오기
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print("x_train :", x_train.shape)
# print("y_train :", y_train.shape)
# print("x_test  :", x_test.shape)
# print("y_test  :", y_test.shape)

# model.summary()


#1. 데이터 준비

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()


print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   #(10000, 32, 32, 3) (10000, 1) #(10000, 32, 32, 3) (10000, 1)

print(np.max(x_train), np.min(x_train)) #255 0 #제일 밝은 값과 제일 어두운 값
print(np.max(x_test), np.min(x_test))   #255 0  #0=검정 255=흰색

#model.summary()

#exit()

##### 스케일링 1 >>>>>>>>>>>0~255 → 0~1 변경 ###########
x_train = x_train/255.  #점찍으면 플로트(실수)표현
x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) #1.0 0.0 #0→0.0=⚫검정
print(np.max(x_test), np.min(x_test))   #1.0 0.0 #255→1.0=⚪흰색

# ####스케일링 2 >>>>>>>>>>>>>0~255 → -1~1 변경 #############
# x_train = (x_train-127.5)/127.5
# x_test = (x_test-127.5)/127.5
# print(np.max(x_train), np.min(x_train)) #1.0 -1.0
# print(np.max(x_test), np.min(x_test))   #1.0 -1.0

# x_train = x_train.reshape(-1, 32,32,1)
# x_test = x_test.reshape(-1, 32,32,1)
# print(x_train.shape, x_test.shape)  # (150000, 32, 32, 1) (30000, 32, 32, 1) #(60000, 28, 28, 1) (10000, 28, 28, 1)

#exit()

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)

y_train = y_train.reshape(50000,1)
y_test = y_test.reshape(10000,1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

print(y_train.shape, y_test.shape)

#model.summary()

#exit()
# ============================================================
# 2. Model Construction - DNN
# ============================================================

model = Sequential()

# 28×28×1 image → 784-dimensional vector
model.add(Flatten(input_shape=(32, 32, 3)))

# Fully Connected Layer
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
# Dropout for regularization
model.add(Dropout(0.2))

# Fully Connected Layer
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
# Dropout for regularization
model.add(Dropout(0.2))

# Fully Connected Layer
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
# Dropout for regularization
model.add(Dropout(0.2))

# Fully Connected Layer
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
# Output Layer - 10 classes
model.add(Dense(100, activation='softmax'))

# Model Summary
model.summary()

# #2. 모델구성
# model = Sequential()
# model.add(Conv2D(64,(3,3), input_shape=(32,32,3)))  #26,26,64
# model.add(Conv2D(filters=32, kernel_size=(3,3),activation='relu'))  #(24,24,32)#filters=32  channels
# model.add(Dropout(0.2))
# model.add(Conv2D(32,(2,2,), activation='relu')) #(23,23,32)
# model.add(Conv2D(16,(2,2,), activation='relu')) #(22,22,16)
# model.add(Dropout(0.2))
# #model.add(Conv2D(32,(2,2,), activation='relu')) #(21,21,32)
# model.add(Dropout(0.2))
# model.add(Conv2D(16,(2,2,), activation='relu')) #(20,20,16) #6400
# model.add(Flatten())                        #납작하게 펴기📦➡️📏

# model.add(Dense(units=32,activation='relu')) #👉 뉴런 32개를 만들겠다
# model.add(Dropout(0.2))
# model.add(Dense(units=16, input_shape =(32,),activation='relu'))
# model.add(Dense(100,activation='softmax'))   #(10,)

# model.summary()

# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 30, 30, 64)        1792      
                                                                 
#  conv2d_1 (Conv2D)           (None, 28, 28, 32)        18464     
                                                                 
#  dropout (Dropout)           (None, 28, 28, 32)        0         
                                                                 
#  conv2d_2 (Conv2D)           (None, 27, 27, 32)        4128      
                                                                 
#  conv2d_3 (Conv2D)           (None, 26, 26, 16)        2064      
                                                                 
#  dropout_1 (Dropout)         (None, 26, 26, 16)        0         
                                                                 
#  conv2d_4 (Conv2D)           (None, 25, 25, 32)        2080      
                                                                 
#  dropout_2 (Dropout)         (None, 25, 25, 32)        0         
                                                                 
#  conv2d_5 (Conv2D)           (None, 24, 24, 16)        2064      
                                                                 
#  flatten (Flatten)           (None, 9216)              0         
                                                                 
#  dense (Dense)               (None, 32)                294944    
                                                                 
#  dropout_3 (Dropout)         (None, 32)                0         
                                                                 
#  dense_1 (Dense)             (None, 16)                528       
                                                                 
#  dense_2 (Dense)             (None, 100)               1700      
                                                                 
# =================================================================
# Total params: 327,764
# Trainable params: 327,764
# Non-trainable params: 0

# exit()

# # #2. 모델구성

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import (
#     Conv2D, MaxPooling2D,
#     BatchNormalization,
#     Dropout,
#     Flatten,
#     Dense
# )

# model = Sequential()

# # ============================================================
# # Block 1
# # ============================================================
# model.add(Conv2D(
#     32, (3,3),
#     padding='same',
#     activation='relu',
#     input_shape=(32,32,3)
# ))

# model.add(BatchNormalization())

# model.add(Conv2D(
#     32, (3,3),
#     padding='same',
#     activation='relu'
# ))

# model.add(BatchNormalization())

# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))


# # ============================================================
# # Block 2
# # ============================================================
# model.add(Conv2D(
#     64, (3,3),
#     padding='same',
#     activation='relu'
# ))

# model.add(BatchNormalization())

# model.add(Conv2D(
#     64, (3,3),
#     padding='same',
#     activation='relu'
# ))

# model.add(BatchNormalization())

# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))


# # ============================================================
# # Classifier
# # ============================================================
# model.add(Flatten())

# model.add(Dense(128, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.3))

# model.add(Dense(100, activation='softmax'))

# model.summary()




#3. 컴파일 , 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam',
            metrics = ['acc'],
            )   

start_time = time.time()
model.fit(x_train,y_train, epochs=100, batch_size=128,
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

########################################################################################################

# loss: 3.146451234817505
# acc: 0.22470000386238098

# 313/313 [==============================] - 0s 1ms/step
# accuracy_score :  0.2247
########################################################################################################
# loss: 1.99714994430542
# acc: 0.48089998960494995
# 313/313 [==============================] - 0s 1ms/step
# accuracy_score :  0.4809
# 걸린시간 :  306.54 초
###################################################  DNN   ##############################################
# loss: 3.401137113571167
# acc: 0.18649999797344208
# 313/313 [==============================] - 0s 945us/step
# accuracy_score :  0.1865
# 걸린시간 :  124.65 초