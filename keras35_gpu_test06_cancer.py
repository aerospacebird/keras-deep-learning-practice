# Classification


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_breast_cancer  #유방암관련 데이터셋을 불러오기

#1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)
print(datasets.feature_names)

#x = datasets.data
x = datasets['data']
y = datasets.target

print(x.shape, y.shape)    #(569, 30) (569,)
print(type(x))  # <class 'numpy.ndarray'>
print(y)
# 0과 1의 갯수를 몇개인지 찾아보기._numpy

print(pd.DataFrame(y).value_counts())

"""
1    357
0    212
Name: count, dtype: int64
[0 1]
(array([0, 1]), array([212, 357]))

"""

print(pd.Series(y).value_counts)
"""
1    357
0    212
Name: count, dtype: int64
"""
print(np.unique(y))  # [0 1]
print(np.unique(y, return_counts=True))  # (array([0, 1]), array([212, 357]))

"""  #  block comments
#x_train, x_test, y_train, y_test = train_test_split(x, y,
                 train_size=0.75,
                 #test_size=0.25,
                 shuffle=True,   # 디폴트 섞는다. 75%/25% 비율로 섞는다.
                 random_state=500,   # 42  디폴트 값이다. 이값을 변경해도 성능에 영향을 줄수가 있다.                
)


""" # block comments
x_train, x_test, y_train, y_test = train_test_split(x, y,
                 train_size=0.75,
                 #test_size=0.25,
                 shuffle=True,   # 디폴트 섞는다. 75%/25% 비율로 섞는다.
                 random_state=337, # # 42  디폴트 값이다. 이값을 변경해도 성능에 영향을 줄수가 있다.
                 stratify=y #훈련 데이터와 테스트 데이터를 나눌 때, y의 클래스 비율이 원본 데이터와 최대한 동일하게 유지되도록 분할하는 옵션입니다.               
)
print(x_train.shape, x_test.shape) # (398, 30)  (171, 30)
print(y_train.shape, y_test.shape) # (398, )    (171, )

####################################################################################################################

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = MaxAbsScaler()
scaler = RobustScaler()

#scaler.fit(x_train)  #준비
#x_train = scaler.transform(x_train)  #변환

x_train = scaler.fit_transform(x_train)  #변환
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))
#0.0 1.0000000000000002
#-0.09792843691148767 1.3768760084020821
#exit()
#######################################################################################################################
#print(np.unique(y_train, return_counts=True))
#print(np.unique(y_test, return_counts=True))


#2. 모델구성
model = Sequential()
model.add(Dense(30, input_dim=30, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(240, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 시그모이드 함수는 입력된 모든 실수를 0과 1 사이의 값으로 변환하여 부드러운 
                                          #  S자 형태(Sigmoid curve)로 출력하는 수학 함수입니다.


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor= 'val_loss', mode='min',
                   patience=20,
                   restore_best_weights=True,
                   verbose=1
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

path= './_save/keras32/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filename = "".join([path, "k32_", date, "-" ,filename])

# 내가 생각하는 파일명
# './save/keras30'+ "k30_"+ "0914_1147"+ "530-0.001.keras"


#exit()


mcp= ModelCheckpoint(
    monitor='val_loss',
    mode= 'auto',
    save_best_only=True,
    filepath=path + 'keras32_mcp6.keras',
    verbose=1
      # loss도 상관없다.
)

start_time = time.time()  #시작시간 반환
hist = model.fit(x_train, y_train, epochs=100, batch_size=4, 
          verbose=1,
          #validation_data = (x_val,y_val),
          callbacks= [es,mcp],
          #verbose=1
          )  # verbose=0 훈련과정을 보여주지 않고 결과만 보여준다. 
end_time = time.time()  # 끝 시간 반환
# verbose=1은 Keras/TensorFlow에서 모델 학습 과정의 진행 상황을 화면에 출력하라는 설정입니다.  verbose=1 default value, 
# verbose=0 : 침묵(아무것도 시현이 않됨)
# verbose=2 : 프로그래스바 삭제,
# verbose=3 : epochs만 나옴,
# verbose= 나머지 : epochs만 나옴.


#model.save(path + 'keras29_3_save_model.keras')   #  훈련이 끝난 상태로 저장하여, 성능이 어느정도 보장된 상태로 저장하였다.
#model.save_weights(path + 'keras29_5_save_weights2.h5')   #  훈련이 끝난 상태로 저장하여, 성능이 어느정도 보장된 상태로 저장하였다.
#model.save_weights(path + 'keras29_5_save_weights2.h5')   #  훈련이 끝난 상태로 저장하여, 성능이 어느정도 보장된 상태로 저장하였다.

#exit()

#4. 평가 및 추론

loss = model.evaluate(x_test, y_test)
print('loss:', loss[0])
print('acc:', round(loss[1], 4))
print("#######################################################################")

y_pred = model.predict(x_test)
print('y_pred:', y_pred)
#print(y_pred[:10])
#result = model.predict()
y_pred = np.round(y_pred) # 0, 1의 결과를 추출하기 위해서, ROUND함수를 사용한다.
print(y_pred[:10])

from sklearn.metrics import accuracy_score
acc_score= accuracy_score(y_test, y_pred)
print("acc_score :", acc_score)


#loss: 0.12190856039524078
#acc: 0.958


#loss: 0.18406589329242706  # StandardScaler
#acc: 0.951                 # StandardScaler

#acc_score : 0.951048951048951 #MaxAbsScaler

#2026-09-15 13:52:09.354077   GPU TIME
#2026-09-15 13:54:54.727495   CPU TIME