# ============================================================
# Conv1D Regression + 6가지 평가 지표
# ============================================================

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ============================================================
# 1. 데이터
# ============================================================

x = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7],
    [6, 7, 8],
    [7, 8, 9],
    [8, 9, 10],
    [9, 10, 11],
    [10, 11, 12],
    [20, 30, 40],
    [30, 40, 50],
    [40, 50, 60]
])

y = np.array([
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    50, 60, 70
])


print("x.shape :", x.shape)
print("y.shape :", y.shape)


# ============================================================
# 2. Conv1D 입력 형태
# ============================================================

x = x.reshape(x.shape[0], x.shape[1], 1)

print("Conv1D x.shape :", x.shape)


# ============================================================
# 3. Conv1D 모델
# ============================================================

model = Sequential()

model.add(
    Conv1D(
        filters=10,
        kernel_size=2,
        activation='relu',
        input_shape=(3, 1)
    )
)

model.add(
    Conv1D(
        filters=10,
        kernel_size=2,
        activation='relu'
    )
)

model.add(Flatten())

model.add(Dense(10, activation='relu'))

model.add(Dense(10, activation='relu'))

model.add(Dense(1))


# ============================================================
# 4. Compile
# ============================================================

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)


# ============================================================
# 5. Callback
# ============================================================

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6
)


# ============================================================
# 6. 학습
# ============================================================

hist = model.fit(
    x,
    y,
    epochs=500,
    batch_size=2,
    validation_split=0.2,
    callbacks=[
        early_stopping,
        reduce_lr
    ],
    verbose=1
)


# ============================================================
# 7. 예측
# ============================================================

pred = model.predict(x, verbose=0).reshape(-1)

print("\n========================================")
print("실제값 vs 예측값")
print("========================================")

for i in range(len(y)):
    print(
        f"{i+1:2d} : "
        f"실제값 = {y[i]:8.2f}   "
        f"예측값 = {pred[i]:8.2f}   "
        f"오차 = {abs(y[i]-pred[i]):8.2f}"
    )


# ============================================================
# 8. MAE
# ============================================================

mae = np.mean(np.abs(y - pred))


# ============================================================
# 9. MSE
# ============================================================

mse = np.mean((y - pred) ** 2)


# ============================================================
# 10. RMSE
# ============================================================

rmse = np.sqrt(mse)


# ============================================================
# 11. R²
# ============================================================

ss_res = np.sum((y - pred) ** 2)

ss_tot = np.sum((y - np.mean(y)) ** 2)

r2 = 1 - (ss_res / ss_tot)


# ============================================================
# 12. ±5 정확도
# ============================================================

accuracy_5 = np.mean(
    np.abs(y - pred) <= 5
) * 100


# ============================================================
# 13. ±10 정확도
# ============================================================

accuracy_10 = np.mean(
    np.abs(y - pred) <= 10
) * 100


# ============================================================
# 14. 최종 평가 결과
# ============================================================

print("\n")
print("================================================")
print("           Conv1D 최종 평가 결과")
print("================================================")

print(f"MAE          : {mae:.4f}")
print(f"MSE          : {mse:.4f}")
print(f"RMSE         : {rmse:.4f}")
print(f"R²           : {r2:.4f}")
print(f"±5 정확도    : {accuracy_5:.2f}%")
print(f"±10 정확도   : {accuracy_10:.2f}%")

print("================================================")

# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, RNN, SimpleRNN, GaussianDropout
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, SimpleRNN

# # 1. 데이터
# x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
#               [5,6,7], [6,7,8], [7,8,9], [8,9,10],
#               [9,10,11], [10,11,12],
#               [20,30,40], [30,40,50], [40,50,60]])
# y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

# print("x.shape :", x.shape)  # (13, 3)
# print("y.shape :", y.shape)  # (13,)

# # RNN 입력 형태로 reshape: (samples, timesteps, features)
# x = x.reshape(x.shape[0], x.shape[1], 1)
# print("RNN 입력 x.shape :", x.shape)  # (13, 3, 1)

# # 2. 모델 구성
# model = Sequential()

# model = Sequential()
#     model.add(Conv1D(filter=10, kernel_size=2, input_shape=(3,1)))
#     model.add(Conv1D(10, 2))
#     model.add(Flatten())
#     model.add(Dense(10, activation='relu'))
#     model.add(Dense(10, activation='relu'))
#     model.add(Dense(1, activation='relu'))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, verbose=1)  # 에포크를 적당히 줄임 (필요시 조정)

# 4. 평가, 예측
result = model.evaluate(x, y, verbose=0)
print('loss :', result)

x_predict = np.array([8, 9, 10]).reshape(-1, 3, 1)
y_predict = model.predict(x_predict, verbose=0)
print('[8, 9, 10]의 예측 결과:', y_predict)

model.summary()

#model.summary()

########################################################################################################
# loss : 0.06630728393793106
# [50, 60, 70]의 예측 결과: [[71.42037]]
