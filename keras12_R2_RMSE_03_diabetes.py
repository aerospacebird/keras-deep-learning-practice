#import ssl
#ssl._create_default_https_context = ssl._create_default_https_context   
from sklearn.datasets import fetch_california_housing, load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)  #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    random_state=42,
)

#2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=10, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=8)

print("========== ========== ========== ========== ==========")

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss :", loss)  #loss : 20.312179565429688


y_predict = model.predict(x_test)   # result = y_predict
from sklearn.metrics import r2_score,mean_squared_error
r2 = r2_score(y_test, y_predict)

print("r2 :", r2)   # r2 : 0.8068209682122945   number one of classroom!!!

mse = mean_squared_error(y_test, y_predict)
print("rmse:", rmse)
def RMSE(y_test, y_predict):  # RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)