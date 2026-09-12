from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))


#실습  train_test_split로 잘라 보시오?

# from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                 train_size=0.75,
                 #test_size=0.3,
                 shuffle=True,   # 디폴트 섞는다. 75%/25% 비율로 섞는다.
                 random_state=500,   # 42  디폴트 값이다. 이값을 변경해도 성능에 영향을 줄수가 있다.                
)


# ==================================================
# Train / Validation 분리
# ==================================================
# 현재 train 데이터의 25%를 validation으로 사용

x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    train_size=0.75,
    shuffle=True,
    random_state=500
)
# Validation을 만들려면 **두 번째 train_test_split()**을 사용해야 합니다.


#2. 모델 
model = Sequential()
model.add(Dense(1, input_dim=1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=3000, batch_size=4, 
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
