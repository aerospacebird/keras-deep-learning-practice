#import ssl
#ssl._create_default_https_context = ssl._create_default_https_context   
from sklearn.datasets import fetch_california_housing, load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split
import numpy as np
import time

#1. 데이터
#datasets = load_diabetes()
#x = datasets.data
#y = datasets.target

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape) #(404, 13) (102, 13)
print(y_train.shape, y_test.shape) #(404,) (102,)

# ==================================================
# Train / Validation 분리
# ==================================================
# 현재 train 데이터의 25%를 validation으로 사용

from sklearn.preprocessing import MinMaxScaler,StandardScaler, MaxAbsScaler, RobustScaler
#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = MaxAbsScaler()
scaler = RobustScaler()

#scaler.fit(x_train)  #준비
#x_train = scaler.transform(x_train)  #변환

x_train = scaler.fit_transform(x_train)  #변환
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))
#0.0 1.0000000000000002
#-0.0019120458891013214 1.1478180091225068
#exit()

x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    train_size=0.75,
    shuffle=True,
    random_state=500
)
# Validation을 만들려면 **두 번째 train_test_split()**을 사용해야 합니다.

#x_train, x_test, y_train, y_test = train_test_split(         데이터 세트가 이미 나눠져 있어서 불필요하다.
    #x, y,
#    random_state=42,
#)

#2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=13, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련

from tensorflow.keras.optimizers import Adam
learning_rate = 0.001

# learnig_rate = 0.001
# learnig_rate = 0.0001
# learnig_rate = 0.005
# learnig_rate = 0.05
# learnig_rate = 0.009


model.compile(loss='mse', optimizer= Adam(learning_rate= learning_rate))

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=40,
    verbose=1,
    restore_best_weights=True,
)
rlr=ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5,
)



start_time = time.time()  #시작시간 반환
hist = model.fit(x_train, y_train,
                 epochs=500, 
                 batch_size=32, 
                 verbose=1,
                 validation_data = (x_val, y_val),
                 callbacks=[es, rlr],
          )  # verbose=0 훈련과정을 보여주지 않고 결과만 보여준다. 
end_time = time.time()  # 끝 시간 반환
# verbose=1은 Keras/TensorFlow에서 모델 학습 과정의 진행 상황을 화면에 출력하라는 설정입니다.  verbose=1 default value, 
# verbose=0 : 침묵(아무것도 시현이 않됨)
# verbose=2 : 프로그래스바 삭제,
# verbose=3 : epochs만 나옴,
# verbose= 나머지 : epochs만 나옴.

#4 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss :', loss) #loss : 4.078690052032471     loss : 20.689849853515625,  loss : 18.26289939880371--->>  StandardScaler 사용하여 개선함.


# loss : 22.485658645629883  RobustScaler
# loss: 1.3795 - val_loss: 15.3487
# loss : 19.564393997192383

print("########################################################## history #########################################################")
print(hist)
print("########################################################## history #########################################################")
print(hist.history)
print("########################################################## loss #########################################################")
print(hist.history['loss'])

print("######################################################### val_loss #########################################################")
print(hist.history['val_loss'])

print("#############################################################################################################################")

      
# 결과는 Dictionary형태로 나온다.

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'  # 캘리포니아 한글 글꼴 깨짐현상 수정
plt.rcParams['axes.unicode_minus'] = False     # 캘리포니아 한글 글꼴 깨짐현상 수정

plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'][3:], c='red', label= 'loss') # y값만 넣으면 시간순으로 그려줌
plt.plot(hist.history['val_loss'][3:], c='blue', label= 'val_loss')
#plt.xlim()
#plt.ylim()
plt.legend(loc='upper right')
plt.title('Boston loss')  # 캘리포니아 한글 글꼴 깨짐현상 수정
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()  # 격자 표시 추가
plt.show()