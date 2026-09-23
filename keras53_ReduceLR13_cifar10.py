#from keras.datasets import fashion_mnist, cifar10

from keras.datasets import cifar10

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
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1) 
print(x_test.shape, y_test.shape)   #(10000, 32, 32, 3) (10000, 1) 

print(np.max(x_train), np.min(x_train)) #255 0 #제일 밝은 값과 제일 어두운 값
print(np.max(x_test), np.min(x_test))   #255 0  #0=검정 255=흰색

#model.summary()


##### 스케일링 1 >>>>>>>>>>>0~255 → 0~1 변경 ###########
x_train = x_train/255.  #점찍으면 플로트(실수)표현
x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) #1.0 0.0 #0→0.0=⚫검정
print(np.max(x_test), np.min(x_test))   #1.0 0.0 #255→1.0=⚪흰색

# ####스케일링 2 >>>>>>>>>>>>>0~255 → -1~1 변경 #############  127.5는 중간값이기 때문이다.
# x_train = (x_train-127.5)/127.5
# x_test = (x_test-127.5)/127.5
#############################################################################################################
# #왜 127.5를 빼는가?
# 이 부분이 매우 중요합니다.

# 원래 범위가

# $$ 0 \sim 255 $$

# 입니다.

# 중간값은

# 2
# 0+255
# 	​

#=127.5
##############################################################################################################

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

#exit()

#2. 모델구성
model = Sequential()
model.add(Conv2D(64,(3,3), input_shape=(32,32,3)))  #26,26,64
model.add(Conv2D(filters=32, kernel_size=(3,3),activation='relu'))  #(24,24,32)  #filters =32  channels
model.add(Dropout(0.2))
model.add(Conv2D(32,(2,2,), activation='relu')) #(23,23,32)
model.add(Conv2D(16,(2,2,), activation='relu')) #(22,22,16)
model.add(Dropout(0.2))
model.add(Conv2D(32,(2,2,), activation='relu')) #(21,21,32)  #32? 불필요?
model.add(Dropout(0.2))
model.add(Conv2D(16,(2,2,), activation='relu')) #(20,20,16) #6400
model.add(Flatten())                        #납작하게 펴기📦➡️📏

model.add(Dense(units=32,activation='relu')) #👉 뉴런 32개를 만들겠다
model.add(Dropout(0.2))
model.add(Dense(units=16, input_shape =(32,),activation='relu'))
model.add(Dense(10,activation='softmax'))   #(10,)

model.summary()
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


from tensorflow.keras.optimizers import Adam
learning_rate = 0.001

# learnig_rate = 0.001
# learnig_rate = 0.0001
# learnig_rate = 0.005
# learnig_rate = 0.05
# learnig_rate = 0.009


#model.compile(loss='mse', optimizer= Adam(learning_rate= learning_rate))


# model.compile(
#     loss='binary_crossentropy',  #  loss함수가 binary_crossentropy
#     optimizer= Adam(learning_rate= learning_rate),
#     metrics=['acc']
# )
#odel.compile(loss='categorical_crossentropy', optimizer= Adam(learning_rate= learning_rate), metrics=['acc']) 
model.compile(loss='categorical_crossentropy', optimizer= Adam(learning_rate= learning_rate),
              metrics = ['acc'],
)
# model.compile(loss='categorical_crossentropy', optimizer = 'adam',
#             metrics = ['acc'],
#             )   
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=40,
    verbose=1,
    restore_best_weights=True,
)
rlr=ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5,
)

start_time = time.time()
model.fit(x_train,y_train, epochs=100, batch_size=128,
        verbose = 1, 
        validation_split = 0.2,
        callbacks=[es, rlr],
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

###########################################################################################################
# loss: 1.0775567293167114
# acc: 0.6549999713897705
# 313/313 [==============================] - 0s 1ms/step
# accuracy_score :  0.655
# 걸린시간 :  310.75 초