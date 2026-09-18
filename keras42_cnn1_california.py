# 30-1 카피
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D,Flatten, GlobalAveragePooling2D #💛💛💛💛💛
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

#1. 데이터 
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
 
x_train, x_test, y_train, y_test = train_test_split(
    x,y ,train_size= 0.75
     , random_state=4333
     )

# x_train, x_val, y_train, y_val =train_test_split(
#     x_train,y_train,
#     train_size = 0.5
#     , random_state=4333
# )

print(x.shape, y.shape) #(20640,8) (20640,)
print(x_train.shape, y_train.shape) #(7740, 8) (7740,)
print(x_test.shape, y_test.shape) #(5160, 8) (5160,)

# exit()

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.preprocessing import RobustScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

# scaler.fit(x_train) 
# x_train = scaler.transform(x_train) 
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test)   

print(np.min(x_train),np.max(x_train))
print(np.min(x_test),np.max(x_test)) 

# x_train = x_train/255.  #점찍으면 플로트(실수)표현
# x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) #1.0 0.0 #0→0.0=⚫검정
print(np.max(x_test), np.min(x_test))   #1.0 0.0 #255→1.0=⚪흰색

x_train = x_train.reshape(-1,8,1,1)
x_test = x_test.reshape(-1,8,1,1)
print(x_train.shape, x_test.shape) 

# exit()
#2. 모델구성
model = Sequential()
model.add(Conv2D(64,(3,1),activation='relu',input_shape=(8,1,1)))  #26,26,64
model.add(Conv2D(filters=32, kernel_size=(3,1),activation='relu'))  #(24,24,32)
# model.add(Dropout(0.2))
model.add(Conv2D(32,(2,1,), activation='relu')) #(23,23,32)
model.add(Conv2D(16,(2,1,), activation='relu')) #(22,22,16)
model.add(Dropout(0.2))
model.add(Conv2D(32,(2,1,), activation='relu')) #(21,21,32)
# model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())#🧀
model.add(Dense(units=32,activation='relu')) #👉 뉴런 32개를 만들겠다
model.add(Dropout(0.2))
model.add(Dense(units=16,activation='relu'))
model.add(Dense(1,))   #(10,)
model.summary()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #🤎🤎🤎🤎🤎

#3. 컴파일 훈련
model.compile(loss='mse',optimizer = 'adam')

es = EarlyStopping(monitor='val_loss', mode= 'min',
                    patience= 30,
                    restore_best_weights= True,
                    verbose=1, #🤎
                   )

start_time = time.time()
hist = model.fit(x_train,y_train,
                 epochs=1000,batch_size=32,validation_split = 0.2,
                callbacks=[es,],      
                # callbacks=[es,mcp],           #🤎🤎🤎🤎🤎
                verbose=1,
                 )
end_time = time.time()


print("=======================================")
#4. 평가예측 
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)
results = model.predict(x_test)
# print(results)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score , mean_squared_error
r2 =  r2_score(y_test, y_predict)
print ("r2 = :" , r2)

#loss(mse) :34.15455627441406
# r2 = : 0.589704701049621 (0.75이상 )

mse = mean_squared_error(y_test, y_predict) #원값 과 예측값
print("mse :" , mse)

def RMSE(y_test, y_predict):         # RMSE 함수를 정의하기
    return np.sqrt(mean_squared_error(y_test, y_predict))
    
rmse  = RMSE(y_test, y_predict)
print("RMSE : ", rmse)
##############################################################################################################
# r2 = : 0.7403711250944806
# mse : 0.3373248138898087
# RMSE :  0.5807967061630159
# PS C:\study> 

'''
결과값
r2 = : 0.745246898244462
mse : 0.33098992809952466
RMSE :  0.5753172412673938

r2 = : 0.6987254246253787
mse : 0.39143331074002496
RMSE :  0.6256463144141624

cnn
r2 = : -0.8608132634948669
mse : 2.4176759538803876
RMSE :  1.554887762470458
'''





# from tensorflow.keras.layers import Dense
# from tensorflow.keras.models import Sequential
# from sklearn.datasets import fetch_california_housing, load_diabetes
# from sklearn.model_selection import train_test_split
# import numpy as np
# import pandas as pd
# import time
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D,Conv1D
#     #MaxPooling2D,
#     #BatchNormalization,
#     #Dropout,
#     #Flatten,
#     #Dense
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
# from sklearn.preprocessing import RobustScaler
# #1. 데이터
# datasets = fetch_california_housing()
# x = datasets.data
# y = datasets.target
# print(x.shape, y.shape)  # (20640, 8) (20640,)

# # ==================================================
# # 1. Data
# # ==================================================

# datasets = fetch_california_housing()

# x = datasets.data
# y = datasets.target

# print("Original X :", x.shape)   # (20640, 8)
# print("Original y :", y.shape)   # (20640,)


# # # ==================================================
# # # 2. Train / Test split
# # # ==================================================

# # x_train, x_test, y_train, y_test = train_test_split(
# #     x,
# #     y,
# #     train_size=0.75,
# #     shuffle=True,
# #     random_state=500
# # )

# # print("x_train :", x_train.shape)
# # print("x_test  :", x_test.shape)


# # # ==================================================
# # # 3. Scaling
# # # ==================================================

# # scaler = RobustScaler()

# # x_train = scaler.fit_transform(x_train)
# # x_test = scaler.transform(x_test)


# # # ==================================================
# # # 4. Train / Validation split
# # # ==================================================

# # x_train, x_val, y_train, y_val = train_test_split(
# #     x_train,
# #     y_train,
# #     train_size=0.75,
# #     shuffle=True,
# #     random_state=500
# # )

# # print("x_train :", x_train.shape)
# # print("x_val   :", x_val.shape)
# # print("x_test  :", x_test.shape)


# # # ==================================================
# # # 5. Reshape for Conv1D
# # # ==================================================

# # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
# # x_val   = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
# # x_test  = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# # print("CNN input shape")
# # print("x_train :", x_train.shape)
# # print("x_val   :", x_val.shape)
# # print("x_test  :", x_test.shape)


# # # #2. 모델구성
# # # model = Sequential()
# # # model.add(Conv2D(128,(3,3), input_shape=(28,28,1)))  #26,26,64
# # # model.add(Conv2D(filters=32, kernel_size=(3,3),activation='relu'))  #(24,24,32)#filters=32  channels
# # # model.add(Dropout(0.2))
# # # model.add(Conv2D(64,(2,2,), activation='relu')) #(23,23,32)
# # # model.add(Conv2D(32,(2,2,), activation='relu')) #(22,22,16)
# # # model.add(Dropout(0.2))
# # # model.add(Conv2D(32,(2,2,), activation='relu')) #(21,21,32)
# # # model.add(Dropout(0.2))
# # # model.add(Conv2D(16,(2,2,), activation='relu')) #(20,20,16) #6400
# # # model.add(Flatten())                        #납작하게 펴기📦➡️📏

# # # model.add(Dense(units=32,activation='relu')) #👉 뉴런 32개를 만들겠다
# # # model.add(Dropout(0.2))
# # # model.add(Dense(units=16, input_shape =(32,),activation='relu'))
# # # model.add(Dense(1,activation='liner'))   #(10,)

# # # model.summary()
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import (
# #     Conv2D,
# #     MaxPooling2D,
# #     BatchNormalization,
# #     Dropout,
# #     Flatten,
# #     Dense
# # )



# # from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# # ==================================================
# # 1. Data
# # ==================================================

# datasets = fetch_california_housing()

# x = datasets.data
# y = datasets.target

# print("Original X :", x.shape)   # (20640, 8)
# print("Original y :", y.shape)   # (20640,)


# # ==================================================
# # 2. Train / Test split
# # ==================================================

# x_train, x_test, y_train, y_test = train_test_split(
#     x,
#     y,
#     train_size=0.75,
#     shuffle=True,
#     random_state=500
# )

# print("x_train :", x_train.shape)
# print("x_test  :", x_test.shape)

# #exit()

# # ==================================================
# # 3. Scaling
# # ==================================================

# scaler = RobustScaler()

# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# #exit()
# # ==================================================
# # 4. Train / Validation split
# # ==================================================

# x_train, x_val, y_train, y_val = train_test_split(
#     x_train,
#     y_train,
#     train_size=0.75,
#     shuffle=True,
#     random_state=500
# )

# print("x_train :", x_train.shape) #x_train : (15480, 8)
# print("x_val   :", x_val.shape)   #x_test  : (5160, 8)
# print("x_test  :", x_test.shape)  #x_test  : (5160, 8)

# #exit()

# # ==================================================
# # 5. Reshape for Conv1D
# # ==================================================

# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
# x_val   = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
# x_test  = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
# exit()
# print("CNN input shape")
# print("x_train :", x_train.shape) #x_train : (11610, 8, 1)
# print("x_val   :", x_val.shape)   # x_val   : (3870, 8, 1)
# print("x_test  :", x_test.shape)  #x_test  : (5160, 8, 1)
# #exit()


# # ==================================================
# # 6. CNN Model
# # ==================================================

# model = Sequential()

# model.add(
#     Conv1D(
#         filters=64,
#         kernel_size=3,
#         activation='relu',
#         input_shape=(8, 1)
#     )
# )

# model.add(BatchNormalization())

# model.add(
#     Conv2D(
#         filters=32,
#         kernel_size=2,
#         activation='relu'
#     )
# )

# model.add(MaxPooling1D(pool_size=2))

# model.add(Dropout(0.2))

# model.add(Flatten())

# model.add(Dense(64, activation='relu'))

# model.add(Dense(32, activation='relu'))

# # Regression
# model.add(Dense(1))


# # ==================================================
# # 7. Model summary
# # ==================================================
# model.summary()

# exit()
# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# es = EarlyStopping(monitor= 'val_loss', mode='min',
#                    patience=20,
#                    restore_best_weights=True,
#                    verbose=1
#                    )
# ################################################# mcp 세이브 파일명 만들기 ##################################################
# import datetime
# date = datetime.datetime.now()  # 2026-09-14 11:41:15.799024

# print(date)
# print(type(date))  #  <class 'datetime.datetime'>

# date = date.strftime("%m%d_%H%M")
# print(date)
# print(type(date))

# #2026-09-14 11:48:29.645739
# #<class 'datetime.datetime'>
# #0914_1148
# #<class 'str'>

# path= './_save/keras32/'
# filename = '{epoch:04d}-{val_loss:.4f}.keras'
# filename = "".join([path, "k32_", date, "-" ,filename])

# # 내가 생각하는 파일명
# # './save/keras30'+ "k30_"+ "0914_1147"+ "530-0.001.keras"


# #exit()







# mcp= ModelCheckpoint(
#     monitor='val_loss',
#     mode= 'auto',
#     save_best_only=True,
#     filepath=path + 'keras32_mcp1.keras',
#     verbose=1
#       # loss도 상관없다.
# )

# start_time = time.time()  #시작시간 반환
# hist = model.fit(x_train, y_train, epochs=100, batch_size=4, 
#           verbose=1,
#           validation_data = (x_val, y_val),
#           callbacks= [es,mcp],
#           #verbose=1
#           )  # verbose=0 훈련과정을 보여주지 않고 결과만 보여준다. 
# end_time = time.time()  # 끝 시간 반환
# # verbose=1은 Keras/TensorFlow에서 모델 학습 과정의 진행 상황을 화면에 출력하라는 설정입니다.  verbose=1 default value, 
# # verbose=0 : 침묵(아무것도 시현이 않됨)
# # verbose=2 : 프로그래스바 삭제,
# # verbose=3 : epochs만 나옴,
# # verbose= 나머지 : epochs만 나옴.


# #model.save(path + 'keras29_3_save_model.keras')   #  훈련이 끝난 상태로 저장하여, 성능이 어느정도 보장된 상태로 저장하였다.
# #model.save_weights(path + 'keras29_5_save_weights2.h5')   #  훈련이 끝난 상태로 저장하여, 성능이 어느정도 보장된 상태로 저장하였다.
# #model.save_weights(path + 'keras29_5_save_weights2.h5')   #  훈련이 끝난 상태로 저장하여, 성능이 어느정도 보장된 상태로 저장하였다.

# #exit()
# #4 평가, 예측

# loss = model.evaluate(x_test, y_test)
# print('loss :', loss) #loss : 4.078690052032471------------->>>>>>>loss : 0.29608967900276184

# #loss : 0.29725757241249084
# #result = model.predict(x)

# # loss : 0.28472891449928284   StandardScaler 사용함. loss value 개선됨.


# # loss : 4.1513495445251465    # RobustScaler

# # 그래프 그리기
# #import matplotlib.pyplot as plt
# #plt.scatter(x, y)
# #plt.plot(x, result, color='red')
# #plt.show()

# print("########################################################## history #########################################################")
# print(hist)
# print("########################################################## history #########################################################")
# print(hist.history)
# print("########################################################## loss #########################################################")
# print(hist.history['loss'])

# print("######################################################### val_loss #########################################################")
# print(hist.history['val_loss'])

# print("#############################################################################################################################")

# #2026-09-15 13:22:40.757619     CPU
# #2026-09-15 13:34:14.618374     GPU   take longer time consumed
# # 결과는 Dictionary형태로 나온다.

# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Malgun Gothic'  # 캘리포니아 한글 글꼴 깨짐현상 수정
# plt.rcParams['axes.unicode_minus'] = False     # 캘리포니아 한글 글꼴 깨짐현상 수정

# plt.figure(figsize=(9, 6))
# plt.plot(hist.history['loss'][3:], c='red', label= 'loss') # y값만 넣으면 시간순으로 그려줌
# plt.plot(hist.history['val_loss'][3:], c='blue', label= 'val_loss')
# #plt.xlim()
# #plt.ylim()
# plt.legend(loc='upper right')
# plt.title('california(캘리포니아) loss')  # 캘리포니아 한글 글꼴 깨짐현상 수정
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()  # 격자 표시 추가
# plt.show()
