import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
              [9,8,7,6,5,4,3,2,1,0]
              ])  #(3, 10)으로 잘못 생성하였다. 이것을 (10,3) shape으로 변형시켜서 주입하시오
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x = x.T   # transpose

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=3))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(1))

#3. compile, training
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000)


#4. 평가 및 추론
loss = model.evaulate(x, y)
print('손실값 :', loss)

results = model.predict(np.array([10, 1.3, 0]))
print('추론 값 :', results)

