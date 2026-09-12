import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)]).T
y = np.array([[1,2,3,4,5,6,7,8,9,10],
               [10, 9, 8, 7, 6, 5, 4, 3, 2, 1,]]).transpose()

print(x.shape, y.shape)  # (3, 10) (10, 2)


#2. 모델 구성
model = Sequential()
model.add(Dense(9, input_dim=3))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(2))


#3. compile, training
model.compile(loss= 'mse', optimizer = 'adam')
model.fit(x, y, epochs=3000)

#4. 평가 및 추론
loss = model.evaluate(x, y)
print('손실값 :', loss)

results = model.predict(np.array([[10, 31, 211]]))
print('추론값 :', results)   # 추론값 : [[10.979818    0.01448749]]

