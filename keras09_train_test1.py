import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])


x_test = np.array([8,9,10])
y_test = np.array([8,9,10])


#2. 모델 
model = Sequential()
model.add(Dense(1, input_dim=1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=2000, batch_size=4)


#4 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss :', loss)
