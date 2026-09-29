import numpy as np
from tensorflow.keras.models import Sequential

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense
)
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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

# 2. 모델구성
model = Sequential()
model.add(LSTM(units=13, input_shape=(3, 1), return_sequences=True))
model.add(LSTM(5, return_sequences=True))
model.add(LSTM(5, return_sequences=True))  # Flatten 전까지는 return_sequences=True
model.add(Flatten())                       # 3D → 2D, # Flatten에서는 return_sequences=True를 사용하면, error를 유발한다.
model.add(Dense(8))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True, verbose=1)
rlr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=1e-5, verbose=1)

model.fit(x, y,
          epochs=500,
          batch_size=1,
          callbacks=[es, rlr],
          verbose=1)

# 4. 평가, 예측
result = model.evaluate(x, y, verbose=0)
print('loss :', result)

# 예측 (학습 샘플 안에서 선택)
x_predict = np.array([[8, 9, 10],
                      [9, 10, 11],
                      [40, 50, 60]]).reshape(3, 3, 1)

y_predict = model.predict(x_predict, verbose=0)

print('[8, 9, 10] 예측 결과 (실제값 11):', y_predict[0])
print('[9, 10, 11] 예측 결과 (실제값 12):', y_predict[1])
print('[40, 50, 60] 예측 결과 (실제값 70):', y_predict[2])

model.summary()
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ lstm (LSTM)                          │ (None, 3, 13)               │             780 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ lstm_1 (LSTM)                        │ (None, 3, 5)                │             380 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ lstm_2 (LSTM)                        │ (None, 3, 5)                │             220 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ flatten (Flatten)                    │ (None, 15)                  │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense (Dense)                        │ (None, 8)                   │             128 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (None, 1)                   │               9 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 4,553 (17.79 KB)
#  Trainable params: 1,517 (5.93 KB)
#  Non-trainable params: 0 (0.00 B)
#  Optimizer params: 3,036 (11.86 KB)
####################################################################################################
# Epoch 396: early stopping
# Restoring model weights from the end of the best epoch: 346.
# loss : 0.020533494651317596
# [8, 9, 10] 예측 결과 (실제값 11): [10.760758]
# [9, 10, 11] 예측 결과 (실제값 12): [11.814487]
# [40, 50, 60] 예측 결과 (실제값 70): [69.99532]
####################################################################################################
