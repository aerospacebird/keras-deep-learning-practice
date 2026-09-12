#import ssl
#ssl._create_default_https_context = ssl._create_default_https_context   
from sklearn.datasets import fetch_california_housing, load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
#datasets = load_diabetes()
#x = datasets.data
#y = datasets.target

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape) #(404, 13) (102, 13)
print(y_train.shape, y_test.shape) #(404,) (102,)

#x_train, x_test, y_train, y_test = train_test_split(         데이터 세트가 이미 나눠져 있어서 불필요하다.
    #x, y,
#    random_state=42,
#)

#2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=13, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=3000, batch_size=10)

print("========== ========== ========== ========== ==========")

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss :", loss)  #loss : 20.312179565429688