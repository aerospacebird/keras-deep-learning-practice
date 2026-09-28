import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# 1. 데이터
a = np.array(range(1, 11))
size = 5

print("a.shape :", a.shape)   # (10,)

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : i + size]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print("bbb.shape :", bbb.shape)   # (6, 5)

# x : 앞의 4개, y : 마지막 1개
x = bbb[:, :-1]   # (6, 4)
y = bbb[:, -1]    # (6,)

print("x.shape :", x.shape)
print("y.shape :", y.shape)

# RNN 입력 형태로 reshape (samples, timesteps, features)
x = x.reshape(x.shape[0], x.shape[1], 1)   # (6, 4, 1)
print("RNN 입력 x.shape :", x.shape)
exit()
# 2. 모델 구성
model = Sequential()
model.add(SimpleRNN(13, input_shape=(4, 1), return_sequences=True))  # timesteps=4, features=1
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
x_predict = np.array([7, 8, 9, 10]).reshape(-1, 4, 1)
y_predict = model.predict(x_predict, verbose=0)
print('[7, 8, 9, 10]의 예측 결과:', y_predict)

model.summary()
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ simple_rnn (SimpleRNN)               │ (None, 4, 13)               │             195 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ simple_rnn_1 (SimpleRNN)             │ (None, 10)                  │             240 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense (Dense)                        │ (None, 10)                  │             110 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (None, 10)                  │             110 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_2 (Dense)                      │ (None, 10)                  │             110 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_3 (Dense)                      │ (None, 1)                   │              11 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 2,330 (9.11 KB)
#  Trainable params: 776 (3.03 KB)
#  Non-trainable params: 0 (0.00 B)
#  Optimizer params: 1,554 (6.07 KB)
#####################################################################################################
# Restoring model weights from the end of the best epoch: 500.
# loss : 0.0012644347734749317
# [7, 8, 9, 10]의 예측 결과: [[10.20476]]
#####################################################################################################
# Restoring model weights from the end of the best epoch: 484.
# loss : 0.00038947700522840023
# [7, 8, 9, 10]의 예측 결과: [[10.452957]]
####################################################################################################