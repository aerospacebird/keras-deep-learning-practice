# https://www.kaggle.com/competitions/bike-sharing-demand/data    # download 받은 url 위치


import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error



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


x_train, x_test, y_train, y_test = train_test_split(
                                   x, y,
    
                                   random_state=42,
)

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

#3. compile & training
model.compile(loss='mse', optimizer= 'Adam')
model.fit(x_train, y_train, epochs=1000, batch_size=32)

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

####################### submission.csv 만들기 // 칼럼에 값을 넣어준다.##########################################################
print(submission)
y_submit = model.predict(test_csv)



submission['count'] = y_submit
print(submission)
print(submission.shape)

submission.to_csv(path + "submit/" + "submit_0907_1005.csv")

