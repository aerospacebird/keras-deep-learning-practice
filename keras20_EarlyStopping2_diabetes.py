from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time

#1. 데이터
#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)  # (20640, 8) (20640,)


#x_train, x_test, y_train, y_test = train_test_split(
#    x, y,
#    random_state=321,
#)


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


#2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=10, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min', 
    patience=100,  # monitor='val_loss'라면:   val_loss가 최저값을 갱신하지 않는 epoch가 10번 연속 발생하면 학습을 중단한다.
    restore_best_weights=True,

)



start_time = time.time()  #시작시간 반환
hist = model.fit(x_train, y_train,
                 epochs=500, 
                 batch_size=32, 
                 verbose=1,
                 validation_data = (x_val, y_val),
                 callbacks=[es],
          )  # verbose=0 훈련과정을 보여주지 않고 결과만 보여준다. 
end_time = time.time()  # 끝 시간 반환
# verbose=1은 Keras/TensorFlow에서 모델 학습 과정의 진행 상황을 화면에 출력하라는 설정입니다.  verbose=1 default value, 
# verbose=0 : 침묵(아무것도 시현이 않됨)
# verbose=2 : 프로그래스바 삭제,
# verbose=3 : epochs만 나옴,
# verbose= 나머지 : epochs만 나옴.

#4 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss :', loss) #loss : 4.078690052032471

print("########################################################## history #########################################################")
print(hist)
print("########################################################## history #########################################################")
print(hist.history)  
print("########################################################## loss #########################################################")
print(hist.history['loss']) # , 'epochs :', len(hist.history['loss]))

print("######################################################### val_loss #########################################################")
print(hist.history['val_loss'])

print("#############################################################################################################################")

      
# 결과는 Dictionary형태로 나온다.

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 글꼴 깨짐현상 수정
plt.rcParams['axes.unicode_minus'] = False     # 한글 글꼴 깨짐현상 수정

plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'][3:], c='red', label= 'loss') # y값만 넣으면 시간순으로 그려줌
plt.plot(hist.history['val_loss'][3:], c='blue', label= 'val_loss')
#plt.xlim()
#plt.ylim()
plt.legend(loc='upper right')
plt.title('Diabets loss')  # 한글 글꼴 깨짐현상 수정
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()  # 격자 표시 추가
plt.show()
