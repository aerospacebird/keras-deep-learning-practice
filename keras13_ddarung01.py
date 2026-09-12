# https://dacon.io/competitions/open/235576/data

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터
path = './_data/ddarung/'  # 상대경로
#path = "c:/study/_data/ddarung/"  # 절대경로
#path = "c:/study/_data/ddarung/"  # 슬래시 역슬래시 상관없어
#path = "c://study//_data//ddarung/"  # 슬래시 2개도 가능하다.  가독성있게 작성하라.

train_csv = pd.read_csv(path + "train.csv", index_col=0) #1459 rows x 11columns ------->>>>#1459 rows x 10 columns index_col=0 때문이다.
print(train_csv) #1459 rows x 10 columns

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv) #[715 rows x 9 columns]

submission = pd.read_csv(path + "submission.csv", index_col=0)
print(submission) #[715 rows x 1 columns]

print(train_csv.shape) #(1459, 10)
print(test_csv.shape) # (715, 9)
print(submission.shape) # (715, 1)

print(train_csv.columns) #Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
                         #'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
                         #'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
                         # dtype='str')
print(train_csv.info())
print(test_csv.info())

#exit()
###################################결측치 처리 1.  삭제      #################################################
train_csv = train_csv.dropna()  # pandas에서 기능을 제공한다.
print(train_csv)     # [1328 rows x 10 columns]

print(test_csv.info())

#exit()

################################## train_csv를 x와 y로 분리 ###################################################
x = train_csv.drop(['count'], axis=1) # raws=0, columns=1, 칼럼을 삭제한다는 의미
print(x)  #[1328 rows x 9 columns]

y = train_csv['count']
print(y)
print(y.shape) # Name: count, Length: 1328, dtype: float64  (1328,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,

    random_state=42,
)


#2. 모델 구성
model = Sequential()
model.add(Dense(128, input_dim=9, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#3. compile & training
model.compile(loss='mse', optimizer= 'adam')
model.fit(x_train, y_train, epochs=3000, batch_size=16)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss :", loss)  #loss : 20.312179565429688


y_predict = model.predict(x_test)   # result = y_predict
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)

print("r2 :", r2)   # r2 : 0.8068209682122945   number one of classroom!!!

mse = mean_squared_error(y_test, y_predict)
print("mse:", mse)
def RMSE(y_test, y_predict):  # RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

# r2 : 0.7152678054300936
# mse: 2171.817967430233
# RMSE : 46.60276780868528

"""      ### block comments  3개의 연속된 따옴표

하이퍼파라미터 튜닝 방법
random_state
train_size
레이어의 깊이
노드의 갯수
epochs number
batch_size



"""