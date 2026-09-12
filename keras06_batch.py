from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=6) # batch_size=4를 추가가능하고 batch_size default value =32.

#4. 평가, 예측
loss= model.evaluate(x, y)
print('loss : ', loss)
# result = model.predict(np.array([1,2,3,4,5,6]))
# print('6의 예측값:', result)
