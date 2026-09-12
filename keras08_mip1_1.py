import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
#x = np.array([[1,2,3,4,5], [6,7,8,9,10]])
x = np.array([[1,6],[2,7], [3,8], [4,9], [5,10]])
y = np.array([1,2,3,4,5])

print(x.shape)  #(5, 2)
print(y.shape)  #(5,)

# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=2)) # input_dim=2 은 열(칼럼)의 갯수와 동일하다.
model.add(Dense(7))  #, input_dim=2))
model.add(Dense(3))  #, input_dim=2))
model.add(Dense(1))  #, input_dim=2))


# 3. compile, training
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=3)


# 4. 평가, 추론
loss= model.evaluate(x, y)
print('loss :', loss)

results = model.predict(np.array([[6, 11]]))
print('predict 의 예측값 : ', results)                               