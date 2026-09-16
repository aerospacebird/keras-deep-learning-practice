# https://www.kaggle.com/competitions/bike-sharing-demand/data    # download 받은 url 위치


import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time


#1. 데이터
path = './_data/kaggle_bike/'  # 상대경로

#path = "c:/study/_data/ddarung/"  # 절대경로
#path = "c:/study/_data/ddarung/"  # 슬래시 역슬래시 상관없어
#path = "c://study//_data//ddarung/"  # 슬래시 2개도 가능하다.  가독성있게 작성하라.

train_csv = pd.read_csv(path + "train.csv", index_col=0) #1459 rows x 11columns ------->>>>#1459 rows x 10 columns index_col=0 때문이다.
print(train_csv) #10886rows x 11 columns

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv) #[6493 rows x 8 columns]

submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)
print(submission) #[6493 rows x 1 columns]

print(train_csv.shape) #(10886, 11)
print(test_csv.shape) # (6493, 8)
print(submission.shape) # (6493, 1)



print(train_csv.columns) 

print(train_csv.info())
print(test_csv.info())

print(train_csv.describe()) 

################################### 결측치 확인 ###########################################################################
print(train_csv.isna().sum())
print(test_csv.isnull().sum())

###################################### x, y 분리 ###########################################################################

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)   #[10886 rows x 8 columns]

y = train_csv['count']
print(y)    
print(y.shape)    #(10886, )


# from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                 train_size=0.75,
                 #test_size=0.25,
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
    random_state=500,
)
# Validation을 만들려면 **두 번째 train_test_split()**을 사용해야 합니다.
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
#scaler = StandardScaler()
#scaler = MinMaxScaler()
#scaler = MaxAbsScaler()
scaler = RobustScaler()

#scaler.fit(x_train)  #준비
#x_train = scaler.transform(x_train)  #변환

x_train = scaler.fit_transform(x_train)  #변환
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))
#0.0 1.0
#-0.035085687818434456 1.0
#exit()
####################################### submit 물밑작업 #######################################################

print(test_csv.info())

###################################결측치 처리 2.  평균값 넣기     #################################################
#test_csv = test_csv.fillna(test_csv.mean())
#print(test_csv.info())  # (6493, 8)
#print(test_csv.shape)  # (6493, 8)


#exit()
#2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim=8, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))

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
    filepath=path + 'keras32_mcp5.keras',
    verbose=1
      # loss도 상관없다.
)

start_time = time.time()  #시작시간 반환
hist = model.fit(x_train, y_train, epochs=1000, batch_size=4, 
          verbose=1,
          validation_data = (x_val, y_val),
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

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss :", loss)  #loss : 20.312179565429688


y_predict = model.predict(x_test)   # result = y_predict
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)

print("r2 :", r2)   # r2 : 0.8068209682122945   number one of classroom!!!

y_submit = model.predict(test_csv)
# y2_pred = model.predict(test_csv)   ## y_submit = model.predict(test_csv)

mse = mean_squared_error(y_test, y_predict)
print("mse:", mse)
def RMSE(y_test, y_predict):  # RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

# r2 : 0.7152678054300936
# mse: 2171.817967430233
# RMSE : 46.60276780868528
# F1 Score는 주로 분류(Classification), 
# AI/딥러닝 분류 모델에서는 Accuracy + Precision + Recall + F1 + Confusion Matrix를 함께 보는 것이 좋습니다.
# R² Score는 회귀(Regression)에서 사용.
# F1 Score는 분류(Classification) 모델의 성능을 평가하는 지표로, **Precision(정밀도)**과 **Recall(재현율)**을 하나의 값으로 결합한 것입니다.
# 두 값의 **조화평균(Harmonic Mean)**을 사용합니다.


#mse: 23077.69921875
#RMSE : 151.91345963656414

#r2 : 0.3158071041107178   # StandardScaler
#mse: 21797.78125          # StandardScaler
#RMSE : 147.6407167755562  # StandardScaler

#r2 : -1.0805761814117432   # MaxAbsScaler
#mse: 66285.3125
#RMSE : 257.45934145025694
####################### submission.csv 만들기 // 칼럼에 값을 넣어준다.##########################################################
print(submission)
y_submit = model.predict(test_csv)



submission['count'] = y_submit
print(submission)
print(submission.shape)

submission.to_csv(path + "submit/" + "submit_0907_1005.csv")

print("########################################################## history #########################################################")
print(hist)
print("########################################################## history #########################################################")
print(hist.history)
print("########################################################## loss #########################################################")
print(hist.history['loss'])

print("######################################################### val_loss #########################################################")
print(hist.history['val_loss'])

print("#############################################################################################################################")

      
# 결과는 Dictionary형태로 나온다.

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'  # 캘리포니아 한글 글꼴 깨짐현상 수정
plt.rcParams['axes.unicode_minus'] = False     # 캘리포니아 한글 글꼴 깨짐현상 수정

plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'][3:], c='red', label= 'loss') # y값만 넣으면 시간순으로 그려줌
plt.plot(hist.history['val_loss'][3:], c='blue', label= 'val_loss')
#plt.xlim()
#plt.ylim()
plt.legend(loc='upper right')
plt.title('kaggle_bike loss')  # 한글 글꼴 깨짐현상 수정
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()  # 격자 표시 추가
plt.show()