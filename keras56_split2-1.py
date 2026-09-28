# ============================================================
# SimpleRNN 시계열 예측
# size = 4
# ============================================================

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ============================================================
# 1. 원본 데이터
# ============================================================

a = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
], dtype=np.float32).T

# 1. 데이터
#a = np.array(range(1, 11))
size = 5

print("a.shape :", a.shape)   # (10,2)

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : i + size]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)

print("bbb.shape :", bbb.shape)   # (6, 5, 2)

#x= bbb[:, :-1, :]
x= bbb[:, :-1]
#y= bbb[:, :-1,1]
y= bbb[:, -1, -1]



print(x.shape, y.shape) # (6, 4, 2) (6,)
#exit()
# RNN 입력 형태로 reshape (samples, timesteps, features)
x = x.reshape(x.shape[0], x.shape[1], 2)   # (6, 4, 1)
print("RNN 입력 x.shape :", x.shape)

#exit()
# 2. 모델 구성
model = Sequential()
model.add(SimpleRNN(13, input_shape=(4, 2), return_sequences=True))  # timesteps=4, features=1
model.add(SimpleRNN(10))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True, verbose=1)
rlr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=20, min_lr=1e-5, verbose=1)

model.fit(x, y,
          epochs=500,
          batch_size=1,
          callbacks=[es, rlr],
          verbose=1)

# 4. 평가, 예측
result = model.evaluate(x, y, verbose=0)
print('loss :', result)

# 예측 (timesteps=4에 맞춰서 입력)
x_predict = np.array([ [7, 3], [8, 2], [9, 1], [10, 0] ], dtype=np.float32).reshape(-1, 4, 2)
y_predict = model.predict(x_predict, verbose=0)
print('[7, 3], [8, 2], [9, 1], [10, 0] ]의 예측 결과:', y_predict)

model.summary()