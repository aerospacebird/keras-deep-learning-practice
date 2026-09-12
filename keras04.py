from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(100, input_dim=3))
model.add(Dense(5, input_dim=100))
model.add(Dense(1, input_dim=5))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1900)

#4. 평가, 예측
loss= model.evaluate(x, y)
print('loss : ', loss)
result = model.predict(np.array([1,2,3,4,5]))
print('6의 예측값:', result)
