import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN

#1.데이터
dataset = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]
])

y = np.array([4,5,6,7,8,9,10])  

print(x.shape, y.shape) #(7, 3) (7,)


x = x.reshape(x.shape[0],x.shape[1],1)
print(x.shape)

#2. 모델구성
model= Sequential()
# model.add(SimpleRNN(units=10, input_shape=(3,1)))
model.add(SimpleRNN(5, input_shape=(3,1))) #  timestep이 3이라고 해서 가중치를 3배로 만들지는 않습니다.
                                           #  RNN은 같은 가중치를 모든 timestep에서 **공유(shared weights)**하기 때문입니다.
                                           # 3 = timestep
                                           # 1 = feature
                                           # timestep이 3이어도 파라미터는 3배가 되지 않습니다.

#3차원으로 들어가서, 2(1)차원으로 나옴 --바로 Dense와 연결가능
model.add(Dense(7, activation='relu'))

model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=150)

#4. 평가, 예측

result= model.evaluate(x,y)
print('loss :', result)

x_predict= np.array([8,9,10]).reshape(1,3,1)
y_predict = model.predict(x_predict, verbose=0)

print('[8,9,10]의 결과:', y_predict)

model.summary()

#  parameter 갯수 = unit*feature + units*bias + units*units  # RNN에서 Parameter 계산공식

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ simple_rnn (SimpleRNN)               │ (None, 5)                   │              35 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense (Dense)                        │ (None, 7)                   │              42 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (None, 1)                   │               8 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 257 (1.01 KB)
#  Trainable params: 85 (340.00 B)
#  Non-trainable params: 0 (0.00 B)


#  SimpleRNN의 파라미터 수는
#  Parameters=(input_dim×units)+(units×units)+units
# ① 입력 → RNN 가중치 1×5=5
# ② 이전 RNN 출력 → 현재 RNN 가중치 5×5=25
# ③ Bias---->>5
#   따라서 첫번째 파라미터는 5 + 25 + 5 = 35 입니다.


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1500)

#4. 평가, 예측

result= model.evaluate(x,y)
print('loss :', result)

x_predict= np.array([8,9,10]).reshape(1,3,1)
y_predict = model.predict(x_predict, verbose=0)

print('[8,9,10] 결과:', y_predict)

# #############################################  result ################################################
# loss : 0.00029511903994716704
# [8,9,10]의 결과: [[10.413102]]

# ######################################################################################################

model.summary()