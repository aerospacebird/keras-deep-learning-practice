from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

# 실습 8개, 4개, 4개  : train, validation, test로 잘라라는 의미이다.

x_train = x[:8]
y_train = y[:8]

x_val = x[8:12]
y_val = y[8:12]


x_test = x[12:]
y_test = y[12:]

print(x_train.shape, x_val.shape, x_test.shape)  #(8,) (4,) (4,)



#2. 모델 
model = Sequential()
model.add(Dense(1, input_dim=1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=4, 
          verbose=1,
          validation_data = (x_val, y_val),
          )  # verbose=0 훈련과정을 보여주지 않고 결과만 보여준다. 
# verbose=1은 Keras/TensorFlow에서 모델 학습 과정의 진행 상황을 화면에 출력하라는 설정입니다.  verbose=1 default value, 
# verbose=0 : 침묵(아무것도 시현이 않됨)
# verbose=2 : 프로그래스바 삭제,
# verbose=3 : epochs만 나옴,
# verbose= 나머지 : epochs만 나옴.

#4 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss :', loss) #loss : 4.078690052032471



