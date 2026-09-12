
# https://dacon.io/competitions/open/235576/data

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import time

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
    random_state=500,
    
)
# Validation을 만들려면 **두 번째 train_test_split()**을 사용해야 합니다.


####################################### submit 물밑작업 #######################################################

print(test_csv.info())

###################################결측치 처리 2.  평균값 넣기     #################################################
test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())  # (715, 9)
print(test_csv.shape)  # (715, 9)










#exit()
#2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim=9, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min', 
    patience=100,  # monitor='val_loss'라면:   val_loss가 최저값을 갱신하지 않는 epoch가 10번 연속 발생하면 학습을 중단한다.
    restore_best_weights=True,

)



start_time = time.time()  #시작시간 반환
hist = model.fit(x_train, y_train,
                 epochs=500, 
                 batch_size=32, 
                 verbose=1,
                 validation_data = (x_val, y_val),
                 callbacks=[es],
          )  # verbose=0 훈련과정을 보여주지 않고 결과만 보여준다. 
end_time = time.time()  # 끝 시간 반환
# verbose=1은 Keras/TensorFlow에서 모델 학습 과정의 진행 상황을 화면에 출력하라는 설정입니다.  verbose=1 default value, 
# verbose=0 : 침묵(아무것도 시현이 않됨)
# verbose=2 : 프로그래스바 삭제,
# verbose=3 : epochs만 나옴,
# verbose= 나머지 : epochs만 나옴.

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

####################### submission.csv 만들기 // 칼럼에 값을 넣어준다.##########################################################
print(submission)
y_submit = model.predict(test_csv)



submission['count'] = y_submit
print(submission)
print(submission.shape)

submission.to_csv(path + "submit/" + "submit_0904_1141.csv")



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
plt.title('Dcon_ddarung loss')  # 캘리포니아 한글 글꼴 깨짐현상 수정
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()  # 격자 표시 추가
plt.show()