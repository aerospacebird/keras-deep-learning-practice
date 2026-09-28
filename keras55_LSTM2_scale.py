import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, RNN, SimpleRNN, GaussianDropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

# 1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print("x.shape :", x.shape)  # (13, 3)
print("y.shape :", y.shape)  # (13,)

# RNN 입력 형태로 reshape: (samples, timesteps, features)
x = x.reshape(x.shape[0], x.shape[1], 1)
print("RNN 입력 x.shape :", x.shape)  # (13, 3, 1)

# 2. 모델 구성
model = Sequential()
# 첫 번째 SimpleRNN: return_sequences=True 필수 (다음 RNN에 3D 출력을 넘기기 위해)
model.add(SimpleRNN(13, input_shape=(3, 1), return_sequences=True))
# 두 번째 SimpleRNN
model.add(SimpleRNN(10))
# Dense 레이어들
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))  # y가 스칼라이므로 units=1

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, verbose=1)  # 에포크를 적당히 줄임 (필요시 조정)

# 4. 평가, 예측
result = model.evaluate(x, y, verbose=0)
print('loss :', result)

x_predict = np.array([50, 60, 70]).reshape(-1, 3, 1)
y_predict = model.predict(x_predict, verbose=0)
print('[50, 60, 70]의 예측 결과:', y_predict)

model.summary()

model.summary()

########################################################################################################
# loss : 0.06630728393793106
# [50, 60, 70]의 예측 결과: [[71.42037]]
