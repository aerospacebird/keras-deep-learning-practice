#36-2 카피

import pandas as pd
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense,Dropout, Flatten, BatchNormalization
import time
from sklearn.metrics import accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

print(np.max(x_train), np.min(x_train)) #255 0 #제일 밝은 값과 제일 어두운 값
print(np.max(x_test), np.min(x_test))   #255 0  #0=검정 255=흰색
#exit()
##### 스케일링 1 >>>>>>>>>>>0~255 → 0~1 변경 ###########
x_train = x_train/255.  #점찍으면 플로트(실수)표현
x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) #1.0 ~ 0.0 #0→0.0=⚫검정
print(np.max(x_test), np.min(x_test))   #1.0 ~ 0.0 #255→1.0=⚪흰색
#exit()
# ####스케일링 2 >>>>>>>>>>>>>0~255 → -1~1 변경 #############
# x_train = (x_train-127.5)/127.5
# x_test = (x_test-127.5)/127.5
# print(np.max(x_train), np.min(x_train)) #1.0 -1.0
# print(np.max(x_test), np.min(x_test))   #1.0 -1.0

x_train = x_train.reshape(-1, 28,28,1)
x_test = x_test.reshape(-1, 28,28,1)
print(x_train.shape, x_test.shape)  #(60000, 28, 28, 1) (10000, 28, 28, 1)

#exit()

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)

y_train = y_train.reshape(60000,1)
y_test = y_test.reshape(-1,1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

print(y_train.shape, y_test.shape) #(60000, 10) (10000, 10)

#exit()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

# ============================================================
# 2. Model Construction - DNN
# ============================================================

model = Sequential()

# 28×28×1 image → 784-dimensional vector
model.add(Flatten(input_shape=(28, 28, 1)))

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
model.add(Dense(10, activation='softmax'))

# Model Summary
model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  flatten (Flatten)           (None, 784)               0         
                                                                 
#  dense (Dense)               (None, 128)               100480    
                                                                 
#  dropout (Dropout)           (None, 128)               0         
                                                                 
#  dense_1 (Dense)             (None, 64)                8256      
                                                                 
#  dropout_1 (Dropout)         (None, 64)                0         
                                                                 
#  dense_2 (Dense)             (None, 32)                2080      
                                                                 
#  dropout_2 (Dropout)         (None, 32)                0         
                                                                 
#  dense_3 (Dense)             (None, 16)                528       
                                                                 
#  dense_4 (Dense)             (None, 10)                170       
                                                                 
# =================================================================
# Total params: 111,514
# Trainable params: 111,514
# Non-trainable params: 0
# ____________________________________
# #2. 모델구성
# model = Sequential()
# model.add(Dense(64, input_shape=(28*28)))

# model.add(Dense(64, input_shape=(28*28)))  #784     26,26,64
# model.add(Dense(filters=32, kernel_size=(3,3),activation='relu'))  #(24,24,32)#filters=32  channels
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu')) #(23,23,32)
# model.add(Dense(32, activation='relu')) #(22,22,16)
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu')) #(21,21,32)
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu')) #(20,20,16) #6400
# #model.add(Flatten())                        #납작하게 펴기📦➡️📏

# model.add(Dense(activation='relu')) #👉 뉴런 32개를 만들겠다
# model.add(Dropout(0.2))
# model.add(Dense(input_shape =(32,),activation='relu'))
# model.add(Dense(1,activation='softmax'))   #(10,)

# model.summary()

#exit()

# #2. 모델구성
# model = Sequential()
# model.add(Conv2D(128,(3,3), input_shape=(28,28,1)))  #26,26,64
# model.add(Conv2D(filters=32, kernel_size=(3,3),activation='relu'))  #(24,24,32)#filters=32  channels
# model.add(Dropout(0.2))
# model.add(Conv2D(64,(2,2,), activation='relu')) #(23,23,32)
# model.add(Conv2D(32,(2,2,), activation='relu')) #(22,22,16)
# model.add(Dropout(0.2))
# model.add(Conv2D(32,(2,2,), activation='relu')) #(21,21,32)
# model.add(Dropout(0.2))
# model.add(Conv2D(16,(2,2,), activation='relu')) #(20,20,16) #6400
# model.add(Flatten())                        #납작하게 펴기📦➡️📏

# model.add(Dense(units=32,activation='relu')) #👉 뉴런 32개를 만들겠다
# model.add(Dropout(0.2))
# model.add(Dense(units=16, input_shape =(32,),activation='relu'))
# model.add(Dense(10,activation='softmax'))   #(10,)

# # model.summary()
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import (
#     Conv2D,
#     MaxPooling2D,
#     BatchNormalization,
#     Dropout,
#     Flatten,
#     Dense
# )

# # ============================================================
# # 2. Model Construction
# # ============================================================

# model = Sequential()

# # Block 1
# model.add(Conv2D(32, (5, 5), activation='relu',input_shape=(28, 28, 1)))  # (3,3)
# model.add(BatchNormalization())
# model.add(Conv2D(32, (5, 5), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))


# # Block 2
# model.add(Conv2D(64, (5, 5), activation='relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(64, (5, 5), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))


# # Classifier
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.30))
# model.add(Dense(10, activation='softmax'))



# model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 26, 26, 32)        320       
                                                                 
#  batch_normalization (BatchN  (None, 26, 26, 32)       128       
#  ormalization)                                                   
                                                                 
#  conv2d_1 (Conv2D)           (None, 24, 24, 32)        9248      
                                                                 
#  batch_normalization_1 (Batc  (None, 24, 24, 32)       128       
#  hNormalization)                                                 
                                                                 
#  max_pooling2d (MaxPooling2D  (None, 12, 12, 32)       0         
#  )                                                               
                                                                 
#  dropout (Dropout)           (None, 12, 12, 32)        0         
                                                                 
#  conv2d_2 (Conv2D)           (None, 10, 10, 64)        18496     
                                                                 
#  batch_normalization_2 (Batc  (None, 10, 10, 64)       256       
#  hNormalization)                                                 
                                                                 
#  conv2d_3 (Conv2D)           (None, 8, 8, 64)          36928     
                                                                 
#  batch_normalization_3 (Batc  (None, 8, 8, 64)         256       
#  hNormalization)                                                 
                                                                 
#  max_pooling2d_1 (MaxPooling  (None, 4, 4, 64)         0         
#  2D)                                                             
                                                                 
#  dropout_1 (Dropout)         (None, 4, 4, 64)          0         
                                                                 
#  flatten (Flatten)           (None, 1024)              0         
                                                                 
#  dense (Dense)               (None, 128)               131200    
                                                                 
#  batch_normalization_4 (Batc  (None, 128)              512       
#  hNormalization)                                                 
                                                                 
#  dropout_2 (Dropout)         (None, 128)               0         
                                                                 
#  dense_1 (Dense)             (None, 10)                1290      
                                                                 
# =================================================================
# Total params: 198,762
# Trainable params: 198,122
# Non-trainable params: 640
# _________________________________________________________________
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 26, 26, 128)       1280      
                                                                 
#  conv2d_1 (Conv2D)           (None, 24, 24, 32)        36896     
                                                                 
#  dropout (Dropout)           (None, 24, 24, 32)        0         
                                                                 
#  conv2d_2 (Conv2D)           (None, 23, 23, 64)        8256      
                                                                 
#  conv2d_3 (Conv2D)           (None, 22, 22, 32)        8224      
                                                                 
#  dropout_1 (Dropout)         (None, 22, 22, 32)        0         
                                                                 
#  conv2d_4 (Conv2D)           (None, 21, 21, 32)        4128      
                                                                 
#  dropout_2 (Dropout)         (None, 21, 21, 32)        0         
                                                                 
#  conv2d_5 (Conv2D)           (None, 20, 20, 16)        2064      
                                                                 
#  flatten (Flatten)           (None, 6400)              0         
                                                                 
#  dense (Dense)               (None, 32)                204832    
                                                                 
#  dropout_3 (Dropout)         (None, 32)                0         
                                                                 
#  dense_1 (Dense)             (None, 16)                528       
                                                                 
#  dense_2 (Dense)             (None, 10)                170       
                                                                 
# =================================================================
# Total params: 266,378
# Trainable params: 266,378
# Non-trainable params: 0
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
#     input_shape=(28,28,1)
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

# model.add(Dense(10, activation='softmax'))

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

####################### CPU의 소요시간 ##########################################
# loss: 0.047359731048345566
# acc: 0.9902999997138977
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step  
# accuracy_score :  0.9903
# 걸린시간 :  1133.95 초

###################### GPU의 소요시간 ###########################################

# loss: 0.04215260595083237
# acc: 0.9897000193595886
# 313/313 [==============================] - 0s 1ms/step
# accuracy_score :  0.9897
# 걸린시간 :  147.02 초

########################## 모델 변경############################################

# accuracy_score :  0.9951
# 걸린시간 :  4460.9 초

##################################################################################
#loss: 0.02116929367184639
#acc: 0.9959999918937683

#accuracy_score :  0.996
#걸린시간 :  174.28 초

################################################################################
# loss: 0.027808737009763718
# acc: 0.9952999949455261
# 313/313 [==============================] - 0s 1ms/step
# accuracy_score :  0.9953
# 걸린시간 :  179.22 초
############################  DNN  ############################################
# loss: 0.10673671960830688
# acc: 0.980400025844574
# 313/313 [==============================] - 0s 680us/step
# accuracy_score :  0.9804
# 걸린시간 :  72.16 초
################################      BatchNormalization     #################
# loss: 0.0867978185415268
# acc: 0.9811000227928162
# 313/313 [==============================] - 0s 886us/step
# accuracy_score :  0.9811
# 걸린시간 :  119.4 초