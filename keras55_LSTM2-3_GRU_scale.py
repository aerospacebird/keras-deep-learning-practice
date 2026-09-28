import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# 1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40], [30,40,50], [40,50,60]], dtype=np.float32)

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70], dtype=np.float32)

print("x.shape :", x.shape)   # (13, 3)
print("y.shape :", y.shape)   # (13,)

# GRU 입력 형태로 reshape
x = x.reshape(x.shape[0], x.shape[1], 1)
print("GRU 입력 x.shape :", x.shape)  # (13, 3, 1)

# 스케일링
x = x / 100.0
y = y / 100.0

# 2. 모델 구성 (GRU)
model = Sequential()

# 첫 번째 GRU 레이어
model.add(GRU(32, input_shape=(3, 1), return_sequences=True))

# 두 번째 GRU 레이어
model.add(GRU(16))

# Dense 레이어
model.add(Dense(16, activation='tanh'))
model.add(Dense(8, activation='tanh'))
model.add(Dense(1))

# 3. 컴파일
optimizer = Adam(learning_rate=0.01)
model.compile(loss='mse', optimizer=optimizer)

# 콜백
es = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True, verbose=1)
rlr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=20, min_lr=1e-5, verbose=1)

# 4. 훈련
history = model.fit(x, y,
                    epochs=1000,
                    batch_size=1,
                    callbacks=[es, rlr],
                    verbose=1)

# 5. 평가 & 예측
result = model.evaluate(x, y, verbose=0)
print('loss :', result)

# 예측 (스케일 복원)
x_predict = np.array([50, 60, 70], dtype=np.float32).reshape(1, 3, 1) / 100.0
y_predict = model.predict(x_predict, verbose=0)
print('[50, 60, 70]의 예측 결과:', y_predict * 100)

model.summary()
##################################################################################################
# Restoring model weights from the end of the best epoch: 1000.
# loss : 1.1912373338418547e-05
# [50, 60, 70]의 예측 결과: [[75.9994]]