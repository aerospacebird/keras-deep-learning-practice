from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np

tf.random.set_seed(333)
np.random.seed(333)

#1. 데이터

x_data = np.array ([[0,0], [0,1], [1,0], [1,1]])
y_data = np.array([0,1,1,0])

#2. 모델 
# model = SVC(kernel='rbf')
model = Sequential()

model.add(Dense(1, input_dim=2, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])
model.fit(x_data, y_data, epochs=1000)

#4. 평가, 예측
y_predict = model.predict(x_data)

result = model.evaluate(x_data, y_data)
print('acc:', result[1])


acc = accuracy_score(y_data, np.round(y_predict))
print("accuracy_score :", acc)
# print('예측값:', y_predict)
# print('정확도:', result)