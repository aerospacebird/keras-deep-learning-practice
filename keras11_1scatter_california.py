#import ssl
#ssl._create_default_https_context = ssl._create_default_https_context   
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)  # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    random_state=321,
)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=32)

print("========== ========== ========== ========== ==========")

loss = model.evaluate(x_test, y_test)
print("loss :", loss)

#4. 평가 예측
loss = model.evaluate(x, y)
print("loss :", loss)

results = model.predict(np.array(x[0].reshape(1, 8)))
print("california housing의 예측값 :", results)

#result = model.predict(x)

# 그래프 그리기
#import matplotlib.pyplot as plt
#plt.scatter(x, y)
#plt.plot(x, result, color='red')
#plt.show()
