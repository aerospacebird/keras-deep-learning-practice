


"""
01: diabetes
02: california
03: BOSTON
04: dacon_ddarung
05: kaggle bike
06: cancer
07: santander
08: wine
09: fetch_covtype
10: digits

"""


from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time

#1. 데이터
#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)  # (20640, 8) (20640,)


#x_train, x_test, y_train, y_test = train_test_split(
#    x, y,
#    random_state=321,
#)


#실습  train_test_split로 잘라 보시오?

# from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                 train_size=0.75,
                 #test_size=0.3,
                 shuffle=True,   # 디폴트 섞는다. 75%/25% 비율로 섞는다.
                 random_state=500,   # 42  디폴트 값이다. 이값을 변경해도 성능에 영향을 줄수가 있다.                
)


# ==================================================
# Train / Validation 분리
# ==================================================
# 현재 train 데이터의 25%를 validation으로 사용

x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    train_size=0.75,
    shuffle=True,
    random_state=500
)
# Validation을 만들려면 **두 번째 train_test_split()**을 사용해야 합니다.
from sklearn.preprocessing import MinMaxScaler,StandardScaler, MaxAbsScaler, RobustScaler
#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = MaxAbsScaler()
scaler = RobustScaler()

#scaler.fit(x_train)  #준비
#x_train = scaler.transform(x_train)  #변환


x_train = scaler.fit_transform(x_train)  #변환
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))
#0.0 1.0
#-0.0492093985517954 1.0136986301369864
#exit()

#2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=10, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()


#######################################################################################################
"""
#2-2  함수형 모델 구성

input1 = Input(shape=(3,))
dense1 = Dense(10, name= 'ys1')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(9, name='ys2')(drop1)
drop2 = Dropout(0.2)(dense2)
output1= Dense(1)(drop2)
model2= Model(inputs=input1, outputs= output1)

model2.summary()

"""
#2-2  함수형 모델 구성

input1 = Input(shape=(10,))
dense1 = Dense(64, name='ys1')(input1)
dense2 = Dense(32, name='ys1')(dense1)
dense3 = Dense(16, name='ys1')(dense2)
dense4 = Dense(8, name='ys1')(dense3)
output1 = Dense(1, name='ys1')(dense4)

model2 = Model(inputs=input1, outputs=output1)


model2.summary()


exit()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min', 
    patience=100,  # monitor='val_loss'라면:   val_loss가 최저값을 갱신하지 않는 epoch가 10번 연속 발생하면 학습을 중단한다.
    restore_best_weights=True,

)

################################################# mcp 세이브 파일명 만들기 ##################################################
import datetime
date = datetime.datetime.now()  # 2026-09-14 11:41:15.799024

print(date)
print(type(date))  #  <class 'datetime.datetime'>

date = date.strftime("%m%d_%H%M")
print(date)
print(type(date))

#2026-09-14 11:48:29.645739
#<class 'datetime.datetime'>
#0914_1148
#<class 'str'>

path= './_save/keras30/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filename = "".join([path, "k30_", date, "-" ,filename])

# 내가 생각하는 파일명
# './save/keras30'+ "k30_"+ "0914_1147"+ "530-0.001.keras"


#exit()

mcp= ModelCheckpoint(
    monitor='val_loss',
    mode= 'auto',
    save_best_only=True,
    filepath=path + 'keras30_mcp1.keras',
    verbose=1
      # loss도 상관없다.
)

start_time = time.time()  #시작시간 반환
hist = model.fit(x_train, y_train,
                 epochs=500, 
                 batch_size=32, 
                 verbose=1,
                 validation_data = (x_val, y_val),
                 callbacks=[es,mcp],
          )  # verbose=0 훈련과정을 보여주지 않고 결과만 보여준다. 
end_time = time.time()  # 끝 시간 반환
# verbose=1은 Keras/TensorFlow에서 모델 학습 과정의 진행 상황을 화면에 출력하라는 설정입니다.  verbose=1 default value, 
# verbose=0 : 침묵(아무것도 시현이 않됨)
# verbose=2 : 프로그래스바 삭제,
# verbose=3 : epochs만 나옴,
# verbose= 나머지 : epochs만 나옴.

from tensorflow.keras.models import load_model

model = load_model('./_save/keras31/keras31_mcp1.keras')

print("모델 로드 완료!")
model.summary()




#4 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss :', loss) #loss : 4.078690052032471    loss : 2926.40478515625   loss : 3104.556640625 ----->>>StandardScaler 사용함.

# loss : 3099.263427734375  loss : 3096.429931640625 # RobustScaler
# loss : 3200.865966796875


print("########################################################## history #########################################################")
print(hist)
print("########################################################## history #########################################################")
print(hist.history)  
print("########################################################## loss #########################################################")
print(hist.history['loss']) # , 'epochs :', len(hist.history['loss]))

print("######################################################### val_loss #########################################################")
print(hist.history['val_loss'])

print("#############################################################################################################################")

      
# 결과는 Dictionary형태로 나온다.

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 글꼴 깨짐현상 수정
plt.rcParams['axes.unicode_minus'] = False     # 한글 글꼴 깨짐현상 수정

plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'][3:], c='red', label= 'loss') # y값만 넣으면 시간순으로 그려줌
plt.plot(hist.history['val_loss'][3:], c='blue', label= 'val_loss')
#plt.xlim()
#plt.ylim()
plt.legend(loc='upper right')
plt.title('Diabets loss')  # 한글 글꼴 깨짐현상 수정
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()  # 격자 표시 추가
plt.show()














