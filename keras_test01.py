import tensorflow.keras.models import Sequential
import tensorflow.keras.layers import Dense
import numpy as np


#1. Data prepare!

x=np.array([1,2,3,4,5])
y=np.array([1,2,3,4,5])


#2. model compose!

model = Sequential()
model.add(Dense(5), input_dim=1)
model.add(Dense(10), input_dim=5)
model.add(Dense(20), input_dim=10)
model.add(Dense(1), input_dim=20)


#3. compile, training!

model.compile(loss= 'mse', optimizer = 'adam')
model.fit(x, y, epochs=100)


#4. evaluate and prediction

loss = model.evaluate(x, y)
print('loss : ', loss)


