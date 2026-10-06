
# ============================================================
# Jena Climate
# 144개 입력 → 144개 wd(deg) 시계열 예측
# Conv1D + LSTM
# ============================================================

# ============================================================
# 1. 라이브러리
# ============================================================

import numpy as np
import pandas as pd
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)


# ============================================================
# 2. 데이터 로드
# ============================================================

path = './_data/kaggle_jena/'

datasets = pd.read_csv(
    path + 'jena_climate_2009_2016.csv',
    index_col=0
)

print('전체 데이터:', datasets.shape)
print(datasets.head())


# ============================================================
# 3. 입력 X / 정답 y
# ============================================================

# 마지막 144개는 최종 검증용 실제 정답
y_cor = datasets[-144:]['wd (deg)'].to_numpy(
    dtype=np.float32
)

# 마지막 288개는 학습에서 제외
# 그중 마지막 144개는 최종 prediction 입력으로 사용
x_data = datasets[:-288].drop(
    ['wd (deg)'],
    axis=1
).to_numpy(dtype=np.float32)

# 입력 X와 같은 시간대의 wd
y_data = datasets[144:-144]['wd (deg)'].to_numpy(
    dtype=np.float32
)

print()
print('x_data:', x_data.shape)
print('y_data:', y_data.shape)
print('y_cor :', y_cor.shape)


# ============================================================
# 4. Sequence 생성 함수
# ============================================================

size_x = 144
size_y = 144


def split_x(dataset, size):

    aaa = []

    for i in range(len(dataset) - size + 1):

        subset = dataset[i:i + size]

        aaa.append(subset)

    return np.array(aaa)


# ============================================================
# 5. Sequence 생성
# ============================================================

start_time = time.time()

x = split_x(x_data, size_x)
y = split_x(y_data, size_y)

end_time = time.time()

print()
print('x :', x.shape)
print('y :', y.shape)

print(
    '자르는 시간:',
    round(end_time - start_time, 2),
    '초'
)


# ============================================================
# 6. 데이터 확인
# ============================================================

print()
print('X sample shape:', x[0].shape)
print('Y sample shape:', y[0].shape)

print()
print('X 전체 shape = (samples, timesteps, features)')
print('Y 전체 shape = (samples, timesteps)')


# ============================================================
# 7. Train / Validation / Test 분리
# ============================================================

# 시계열 데이터이므로 shuffle하지 않음

total = len(x)

train_end = int(total * 0.8)
val_end = int(total * 0.9)

x_train = x[:train_end]
y_train = y[:train_end]

x_val = x[train_end:val_end]
y_val = y[train_end:val_end]

x_test = x[val_end:]
y_test = y[val_end:]


print()
print('==============================')
print('데이터 분할')
print('==============================')

print('x_train:', x_train.shape)
print('y_train:', y_train.shape)

print('x_val  :', x_val.shape)
print('y_val  :', y_val.shape)

print('x_test :', x_test.shape)
print('y_test :', y_test.shape)


# ============================================================
# 8. Scaling
# ============================================================

# ------------------------------------------------------------
# X scaling
# 13개 feature 각각 독립적으로 scaling
# ------------------------------------------------------------

N_FEATURES = x_train.shape[2]

x_scaler = MinMaxScaler()

x_train_2d = x_train.reshape(-1, N_FEATURES)
x_val_2d   = x_val.reshape(-1, N_FEATURES)
x_test_2d  = x_test.reshape(-1, N_FEATURES)

x_train_2d = x_scaler.fit_transform(x_train_2d)
x_val_2d   = x_scaler.transform(x_val_2d)
x_test_2d  = x_scaler.transform(x_test_2d)

x_train = x_train_2d.reshape(
    x_train.shape
)

x_val = x_val_2d.reshape(
    x_val.shape
)

x_test = x_test_2d.reshape(
    x_test.shape
)


# ------------------------------------------------------------
# y scaling
# ------------------------------------------------------------

y_scaler = MinMaxScaler()

y_train = y_scaler.fit_transform(
    y_train.reshape(-1, 1)
).reshape(y_train.shape)

y_val = y_scaler.transform(
    y_val.reshape(-1, 1)
).reshape(y_val.shape)

y_test = y_scaler.transform(
    y_test.reshape(-1, 1)
).reshape(y_test.shape)


print()
print('Scaling 완료')


# ============================================================
# 9. 모델 구성
# ============================================================

TIMESTEPS = 144
N_FEATURES = 13


model = Sequential()


# ------------------------------------------------------------
# Conv1D
# ------------------------------------------------------------

model.add(
    Conv1D(
        filters=64,
        kernel_size=5,
        padding='same',
        activation='relu',
        input_shape=(TIMESTEPS, N_FEATURES)
    )
)


# ------------------------------------------------------------
# 두 번째 Conv1D
# ------------------------------------------------------------

model.add(
    Conv1D(
        filters=128,
        kernel_size=5,
        padding='same',
        activation='relu'
    )
)


# ------------------------------------------------------------
# Dropout
# ------------------------------------------------------------

model.add(
    Dropout(0.2)
)


# ------------------------------------------------------------
# LSTM
# return_sequences=True
#
# 이유:
# 144개의 시점 각각에 대해
# 하나의 wd 값을 예측해야 하기 때문
# ------------------------------------------------------------

model.add(
    LSTM(
        128,
        return_sequences=True
    )
)


model.add(
    Dropout(0.2)
)


# ------------------------------------------------------------
# 각 timestep마다 1개의 wd(deg) 출력
# ------------------------------------------------------------

model.add(
    Dense(1)
)


model.summary()


# ============================================================
# 10. Compile
# ============================================================

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)


# ============================================================
# 11. Callback
# ============================================================

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)


reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)


checkpoint = ModelCheckpoint(
    './_data/kaggle_jena_npy/best_jena_model.keras',
    monitor='val_loss',
    save_best_only=True
)


# ============================================================
# 12. 모델 학습
# ============================================================

start_time = time.time()


history = model.fit(
    x_train,
    y_train[..., np.newaxis],
    validation_data=(
        x_val,
        y_val[..., np.newaxis]
    ),
    epochs=100,
    batch_size=128,
    shuffle=False,
    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ],
    verbose=1
)


end_time = time.time()


print()
print('===================================')
print('훈련 완료')
print('훈련 시간:', round(end_time - start_time, 2), '초')
print('===================================')


# ============================================================
# 13. Test 평가
# ============================================================

test_loss, test_mae = model.evaluate(
    x_test,
    y_test[..., np.newaxis],
    verbose=0
)


print()
print('===================================')
print('TEST 평가')
print('===================================')

print('Test Loss:', test_loss)
print('Test MAE :', test_mae)


# ============================================================
# 14. Test Prediction
# ============================================================

y_pred_scaled = model.predict(
    x_test,
    verbose=0
)


print()
print('y_pred_scaled:', y_pred_scaled.shape)


# ============================================================
# 15. 원래 단위로 복원
# ============================================================

y_pred = y_scaler.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
).reshape(
    y_pred_scaled.shape[0],
    y_pred_scaled.shape[1]
)


y_test_original = y_scaler.inverse_transform(
    y_test.reshape(-1, 1)
).reshape(
    y_test.shape
)


print()
print('y_pred:', y_pred.shape)
print('y_test:', y_test_original.shape)


# ============================================================
# 16. 평가 지표
# ============================================================

y_true_flat = y_test_original.reshape(-1)
y_pred_flat = y_pred.reshape(-1)


mae = mean_absolute_error(
    y_true_flat,
    y_pred_flat
)


mse = mean_squared_error(
    y_true_flat,
    y_pred_flat
)


rmse = np.sqrt(mse)


r2 = r2_score(
    y_true_flat,
    y_pred_flat
)


# ============================================================
# ±5 / ±10 정확도
# ============================================================

error = np.abs(
    y_true_flat - y_pred_flat
)


accuracy_5 = np.mean(
    error <= 5
) * 100


accuracy_10 = np.mean(
    error <= 10
) * 100


print()
print('==============================================')
print('최종 TEST 성능')
print('==============================================')

print(f'MAE       : {mae:.4f}')
print(f'MSE       : {mse:.4f}')
print(f'RMSE      : {rmse:.4f}')
print(f'R²        : {r2:.4f}')
print(f'±5 정확도 : {accuracy_5:.2f}%')
print(f'±10 정확도: {accuracy_10:.2f}%')


# ============================================================
# 17. 마지막 144개 입력 데이터 생성
# ============================================================

x_predict = datasets[-288:-144].drop(
    ['wd (deg)'],
    axis=1
).to_numpy(
    dtype=np.float32
)


x_predict = x_predict.reshape(
    1,
    144,
    13
)


print()
print('x_predict:', x_predict.shape)


# ============================================================
# 18. x_predict Scaling
# ============================================================

x_predict_scaled = x_scaler.transform(
    x_predict.reshape(-1, 13)
).reshape(
    1,
    144,
    13
)


# ============================================================
# 19. 미래 144개 wd 예측
# ============================================================

y_future_scaled = model.predict(
    x_predict_scaled,
    verbose=0
)


y_future = y_scaler.inverse_transform(
    y_future_scaled.reshape(-1, 1)
).reshape(144)


print()
print('==============================================')
print('미래 144개 wd(deg) 예측')
print('==============================================')

print(y_future)


# ============================================================
# 20. 실제 정답과 비교
# ============================================================

future_error = np.abs(
    y_cor - y_future
)


future_mae = np.mean(
    future_error
)


future_mse = np.mean(
    (y_cor - y_future) ** 2
)


future_rmse = np.sqrt(
    future_mse
)


future_r2 = r2_score(
    y_cor,
    y_future
)


future_acc_5 = np.mean(
    future_error <= 5
) * 100


future_acc_10 = np.mean(
    future_error <= 10
) * 100


print()
print('==============================================')
print('최종 144개 미래 예측 성능')
print('==============================================')

print(f'MAE       : {future_mae:.4f}')
print(f'MSE       : {future_mse:.4f}')
print(f'RMSE      : {future_rmse:.4f}')
print(f'R²        : {future_r2:.4f}')
print(f'±5 정확도 : {future_acc_5:.2f}%')
print(f'±10 정확도: {future_acc_10:.2f}%')


# ============================================================
# 21. 실제값 vs 예측값
# ============================================================

result = pd.DataFrame({
    'Actual_wd': y_cor,
    'Predicted_wd': y_future,
    'Error': future_error
})


print()
print(result.head(20))


# ============================================================
# 22. 저장
# ============================================================

result.to_csv(
    './_data/kaggle_jena_npy/jena_wd_prediction.csv',
    index=False
)


print()
print('예측 결과 저장 완료')
# ============================================================
# CRAZY AI
# Jena Climate 24시간 → 다음 24시간 풍향 예측
# Conv1D 144 → 144 Sequence Prediction
# ============================================================


# # ============================================================
# # 1. 라이브러리
# # ============================================================

# import os
# import time
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import (
#     mean_absolute_error,
#     mean_squared_error,
#     r2_score
# )

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import (
#     Conv1D,
#     BatchNormalization,
#     MaxPooling1D,
#     GlobalAveragePooling1D,
#     Dense,
#     Dropout
# )

# from tensorflow.keras.callbacks import (
#     EarlyStopping,
#     ReduceLROnPlateau,
#     ModelCheckpoint
# )


# # ============================================================
# # 2. 데이터 로드
# # ============================================================

# path = './_data/kaggle_jena/'

# datasets = pd.read_csv(
#     path + 'jena_climate_2009_2016.csv',
#     index_col=0
# )

# print("=" * 70)
# print("1. ORIGINAL DATA")
# print("=" * 70)

# print("datasets.shape :", datasets.shape)
# print("columns        :", datasets.columns.tolist())


# # ============================================================
# # 3. 기본 설정
# # ============================================================

# TIMESTEPS = 144
# PREDICT_STEPS = 144

# N_FEATURES = 13


# # ============================================================
# # 4. 마지막 실제 정답 144개 저장
# #
# # 마지막 144개 = 우리가 최종적으로 예측할 대상
# # ============================================================

# y_cor = datasets[
#     'wd (deg)'
# ].iloc[-PREDICT_STEPS:].to_numpy(
#     dtype=np.float32
# )

# print("\n실제 최종 정답 y_cor")
# print("shape:", y_cor.shape)


# # ============================================================
# # 5. X / Y 데이터 구성
# #
# # X:
# # 전체 데이터에서 마지막 288개 제외
# #
# # Y:
# # 144개 뒤부터 시작하고 마지막 144개 제외
# #
# # 즉,
# #
# # X : t ~ t+143
# # Y : t+144 ~ t+287
# #
# # 과거 24시간 → 미래 24시간
# # ============================================================

# x_data = datasets.iloc[:-288].drop(
#     ['wd (deg)'],
#     axis=1
# ).to_numpy(
#     dtype=np.float32
# )

# y_data = datasets[
#     'wd (deg)'
# ].iloc[144:-144].to_numpy(
#     dtype=np.float32
# )


# print("\n" + "=" * 70)
# print("2. X / Y DATA")
# print("=" * 70)

# print("x_data shape:", x_data.shape)
# print("y_data shape:", y_data.shape)


# # ============================================================
# # 6. Sequence 생성 함수
# # ============================================================

# def split_x(dataset, size):

#     sequences = []

#     for i in range(
#         len(dataset) - size + 1
#     ):

#         subset = dataset[
#             i:i + size
#         ]

#         sequences.append(subset)

#     return np.array(
#         sequences,
#         dtype=np.float32
#     )


# # ============================================================
# # 7. Sequence 생성
# # ============================================================

# print("\n" + "=" * 70)
# print("3. SEQUENCE GENERATION")
# print("=" * 70)

# start_time = time.time()

# x = split_x(
#     x_data,
#     TIMESTEPS
# )

# y = split_x(
#     y_data,
#     PREDICT_STEPS
# )

# end_time = time.time()


# print("x :", x.shape)
# print("y :", y.shape)

# print(
#     "Sequence 생성 시간:",
#     round(end_time - start_time, 2),
#     "sec"
# )


# # 예상
# #
# # x : (420120, 144, 13)
# # y : (420120, 144)


# # ============================================================
# # 8. 최종 예측용 X
# #
# # 마지막 실제 정답 144개 바로 앞의
# # 144개 입력 데이터를 사용
# #
# # x_predict
# # = 실제로 모델에게 마지막으로 넣을 데이터
# # ============================================================

# x_predict = datasets[
#     ['wd (deg)']
# ].copy()


# x_predict = datasets.iloc[
#     -288:-144
# ].drop(
#     ['wd (deg)'],
#     axis=1
# ).to_numpy(
#     dtype=np.float32
# )


# x_predict = x_predict.reshape(
#     1,
#     TIMESTEPS,
#     N_FEATURES
# )


# print("\n최종 예측용 X")
# print("x_predict:", x_predict.shape)

# print("y_cor:", y_cor.shape)


# # ============================================================
# # 9. 시간 순서 기반 Train / Validation / Test
# #
# # 절대로 shuffle하지 않음
# # ============================================================

# total_samples = len(x)

# train_size = int(
#     total_samples * 0.70
# )

# val_size = int(
#     total_samples * 0.15
# )

# test_size = (
#     total_samples
#     - train_size
#     - val_size
# )


# print("\n" + "=" * 70)
# print("4. DATA SPLIT")
# print("=" * 70)

# print("total :", total_samples)
# print("train :", train_size)
# print("val   :", val_size)
# print("test  :", test_size)


# # ============================================================
# # 10. 데이터 분리
# # ============================================================

# x_train = x[
#     :train_size
# ]

# y_train = y[
#     :train_size
# ]


# x_val = x[
#     train_size:
#     train_size + val_size
# ]

# y_val = y[
#     train_size:
#     train_size + val_size
# ]


# x_test = x[
#     train_size + val_size:
# ]

# y_test = y[
#     train_size + val_size:
# ]


# print("\nTrain:")
# print(
#     x_train.shape,
#     y_train.shape
# )

# print("\nValidation:")
# print(
#     x_val.shape,
#     y_val.shape
# )

# print("\nTest:")
# print(
#     x_test.shape,
#     y_test.shape
# )


# # ============================================================
# # 11. Scaling
# #
# # 매우 중요:
# #
# # scaler는 TRAIN DATA로만 fit
# #
# # 이렇게 해야 Validation/Test 정보가
# # 학습 과정에 유입되는 Data Leakage를 방지
# # ============================================================


# # ------------------------------------------------------------
# # X scaler
# # ------------------------------------------------------------

# scaler_x = StandardScaler()

# x_train_2d = x_train.reshape(
#     -1,
#     N_FEATURES
# )

# scaler_x.fit(
#     x_train_2d
# )


# # 전체 X scaling 함수
# def scale_x(data):

#     original_shape = data.shape

#     data_2d = data.reshape(
#         -1,
#         N_FEATURES
#     )

#     data_scaled = scaler_x.transform(
#         data_2d
#     )

#     return data_scaled.reshape(
#         original_shape
#     )


# x_train = scale_x(x_train)
# x_val = scale_x(x_val)
# x_test = scale_x(x_test)
# x_predict = scale_x(x_predict)


# # ------------------------------------------------------------
# # Y scaler
# # ------------------------------------------------------------

# scaler_y = StandardScaler()

# y_train_2d = y_train.reshape(
#     -1,
#     1
# )

# scaler_y.fit(
#     y_train_2d
# )


# def scale_y(data):

#     original_shape = data.shape

#     data_2d = data.reshape(
#         -1,
#         1
#     )

#     data_scaled = scaler_y.transform(
#         data_2d
#     )

#     return data_scaled.reshape(
#         original_shape
#     )


# y_train = scale_y(y_train)
# y_val = scale_y(y_val)
# y_test = scale_y(y_test)


# print("\n" + "=" * 70)
# print("5. SCALING")
# print("=" * 70)

# print("x_train:", x_train.shape)
# print("x_val  :", x_val.shape)
# print("x_test :", x_test.shape)

# print("y_train:", y_train.shape)
# print("y_val  :", y_val.shape)
# print("y_test :", y_test.shape)

# print("x_predict:", x_predict.shape)
# # ======================================================================
# # 5. SCALING
# # ======================================================================
# # x_train: (294084, 144, 13)
# # x_val  : (63018, 144, 13)
# # x_test : (63018, 144, 13)
# # y_train: (294084, 144)
# # y_val  : (63018, 144)
# # y_test : (63018, 144)
# # x_predict: (1, 144, 13)
# #exit()
# # ============================================================
# # 12. Conv1D Model
# # ============================================================

# print("\n" + "=" * 70)
# print("6. CONV1D MODEL")
# print("=" * 70)


# model = Sequential([

#     # --------------------------------------------------------
#     # Input
#     #
#     # (144, 13)
#     #
#     # 144 = 과거 24시간
#     # 13  = 기상 특징
#     # --------------------------------------------------------

#     Conv1D(
#         filters=64,
#         kernel_size=5,
#         padding='same',
#         activation='relu',
#         input_shape=(
#             TIMESTEPS,
#             N_FEATURES
#         )
#     ),

#     BatchNormalization(),

#     Dropout(0.2),


#     # --------------------------------------------------------
#     # Conv1D #2
#     # --------------------------------------------------------

#     Conv1D(
#         filters=128,
#         kernel_size=5,
#         padding='same',
#         activation='relu'
#     ),

#     BatchNormalization(),

#     Dropout(0.2),


#     # --------------------------------------------------------
#     # Conv1D #3
#     # --------------------------------------------------------

#     Conv1D(
#         filters=128,
#         kernel_size=3,
#         padding='same',
#         activation='relu'
#     ),

#     BatchNormalization(),


#     # --------------------------------------------------------
#     # 시간축 압축
#     # --------------------------------------------------------

#     MaxPooling1D(
#         pool_size=2
#     ),


#     # --------------------------------------------------------
#     # Conv1D #4
#     # --------------------------------------------------------

#     Conv1D(
#         filters=64,
#         kernel_size=3,
#         padding='same',
#         activation='relu'
#     ),

#     BatchNormalization(),


#     # --------------------------------------------------------
#     # Global Average Pooling
#     # --------------------------------------------------------

#     GlobalAveragePooling1D(),


#     # --------------------------------------------------------
#     # Dense
#     # --------------------------------------------------------

#     Dense(
#         256,
#         activation='relu'
#     ),

#     Dropout(0.3),


#     Dense(
#         128,
#         activation='relu'
#     ),

#     Dropout(0.2),


#     # --------------------------------------------------------
#     # 미래 144개 예측
#     # --------------------------------------------------------

#     Dense(
#         PREDICT_STEPS
#     )
# ])


# # ============================================================
# # 13. Compile
# # ============================================================

# model.compile(
#     optimizer='adam',
#     loss='mse',
#     metrics=['mae']
# )


# # ============================================================
# # 14. Model Summary
# # ============================================================

# model.summary()


# # ============================================================
# # 15. Callbacks
# # ============================================================

# os.makedirs(
#     './_data/kaggle_jena_model/',
#     exist_ok=True
# )


# es = EarlyStopping(
#     monitor='val_loss',
#     patience=12,
#     restore_best_weights=True,
#     verbose=1
# )


# rlr = ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.5,
#     patience=5,
#     min_lr=1e-6,
#     verbose=1
# )


# mcp = ModelCheckpoint(
#     './_data/kaggle_jena_model/'
#     'jena_conv1d_best.keras',

#     monitor='val_loss',

#     save_best_only=True,

#     verbose=1
# )


# # ============================================================
# # 16. Training
# # ============================================================

# print("\n" + "=" * 70)
# print("7. MODEL TRAINING")
# print("=" * 70)


# history = model.fit(

#     x_train,
#     y_train,

#     validation_data=(
#         x_val,
#         y_val
#     ),

#     epochs=100,

#     batch_size=128,

#     shuffle=False,

#     callbacks=[
#         es,
#         rlr,
#         mcp
#     ],

#     verbose=1
# )


# # ============================================================
# # 17. Test Prediction
# # ============================================================

# print("\n" + "=" * 70)
# print("8. TEST PREDICTION")
# print("=" * 70)


# y_pred_scaled = model.predict(
#     x_test,
#     batch_size=128,
#     verbose=1
# )


# print(
#     "y_pred_scaled:",
#     y_pred_scaled.shape
# )


# # ============================================================
# # 18. Scaling 복원
# # ============================================================

# y_test_orig = scaler_y.inverse_transform(
#     y_test.reshape(-1, 1)
# ).reshape(
#     y_test.shape
# )


# y_pred_orig = scaler_y.inverse_transform(
#     y_pred_scaled.reshape(-1, 1)
# ).reshape(
#     y_pred_scaled.shape
# )


# print(
#     "y_test_orig:",
#     y_test_orig.shape
# )

# print(
#     "y_pred_orig:",
#     y_pred_orig.shape
# )


# # ============================================================
# # 19. Flatten
# #
# # 2차원
# #
# # (samples, 144)
# #
# # → 1차원
# # ============================================================

# y_true_flat = y_test_orig.flatten()

# y_pred_flat = y_pred_orig.flatten()


# # ============================================================
# # 20. MSE
# # ============================================================

# mse = mean_squared_error(
#     y_true_flat,
#     y_pred_flat
# )


# # ============================================================
# # 21. MAE
# # ============================================================

# mae = mean_absolute_error(
#     y_true_flat,
#     y_pred_flat
# )


# # ============================================================
# # 22. RMSE
# # ============================================================

# rmse = np.sqrt(
#     mse
# )


# # ============================================================
# # 23. R²
# # ============================================================

# r2 = r2_score(
#     y_true_flat,
#     y_pred_flat
# )


# # ============================================================
# # 24. ±5° Accuracy
# # ============================================================

# error = np.abs(
#     y_true_flat -
#     y_pred_flat
# )


# accuracy_5 = (
#     np.mean(
#         error <= 5
#     )
#     * 100
# )


# # ============================================================
# # 25. ±10° Accuracy
# # ============================================================

# accuracy_10 = (
#     np.mean(
#         error <= 10
#     )
#     * 100
# )


# # ============================================================
# # 26. Test Performance
# # ============================================================

# print("\n")
# print("=" * 70)
# print("              CONV1D TEST PERFORMANCE")
# print("=" * 70)

# print(
#     f"MSE          : {mse:.6f}"
# )

# print(
#     f"MAE          : {mae:.6f}°"
# )

# print(
#     f"RMSE         : {rmse:.6f}°"
# )

# print(
#     f"R²           : {r2:.6f}"
# )

# print(
#     f"±5° Accuracy  : {accuracy_5:.2f}%"
# )

# print(
#     f"±10° Accuracy : {accuracy_10:.2f}%"
# )

# print("=" * 70)


# # ============================================================
# # 27. 최종 24시간 예측
# #
# # x_predict
# # =
# # 마지막 실제 정답 144개 바로 앞의 144개
# # ============================================================

# print("\n" + "=" * 70)
# print("9. FINAL 24-HOUR PREDICTION")
# print("=" * 70)


# final_pred_scaled = model.predict(
#     x_predict,
#     verbose=0
# )


# final_pred = scaler_y.inverse_transform(
#     final_pred_scaled.reshape(-1, 1)
# ).reshape(
#     final_pred_scaled.shape
# )


# print(
#     "final_pred shape:",
#     final_pred.shape
# )


# # ============================================================
# # 28. 실제값과 예측값 비교
# # ============================================================

# print("\n")
# print("=" * 70)
# print("10. PREDICTION vs ACTUAL")
# print("=" * 70)


# final_pred_1d = final_pred[0]


# for i in range(PREDICT_STEPS):

#     print(
#         f"{i+1:3d} | "
#         f"예측 = {final_pred_1d[i]:8.2f}° | "
#         f"실제 = {y_cor[i]:8.2f}° | "
#         f"오차 = "
#         f"{abs(final_pred_1d[i] - y_cor[i]):8.2f}°"
#     )


# # ============================================================
# # 29. 최종 24시간 예측 MAE
# # ============================================================

# final_mae = mean_absolute_error(
#     y_cor,
#     final_pred_1d
# )


# final_rmse = np.sqrt(
#     mean_squared_error(
#         y_cor,
#         final_pred_1d
#     )
# )


# final_error = np.abs(
#     y_cor -
#     final_pred_1d
# )


# final_acc_5 = (
#     np.mean(
#         final_error <= 5
#     )
#     * 100
# )


# final_acc_10 = (
#     np.mean(
#         final_error <= 10
#     )
#     * 100
# )


# print("\n" + "=" * 70)
# print("11. FINAL 24-HOUR PERFORMANCE")
# print("=" * 70)

# print(
#     f"MAE          : {final_mae:.4f}°"
# )

# print(
#     f"RMSE         : {final_rmse:.4f}°"
# )

# print(
#     f"±5° Accuracy  : {final_acc_5:.2f}%"
# )

# print(
#     f"±10° Accuracy : {final_acc_10:.2f}%"
# )

# print("=" * 70)


# # ============================================================
# # 30. 학습 Loss
# # ============================================================

# plt.figure(
#     figsize=(12, 5)
# )

# plt.plot(
#     history.history['loss'],
#     label='Train Loss'
# )

# plt.plot(
#     history.history['val_loss'],
#     label='Validation Loss'
# )

# plt.title(
#     'Conv1D Train / Validation Loss'
# )

# plt.xlabel(
#     'Epoch'
# )

# plt.ylabel(
#     'MSE'
# )

# plt.legend()

# plt.grid()

# plt.tight_layout()

# plt.show()


# # ============================================================
# # 31. 학습 MAE
# # ============================================================

# plt.figure(
#     figsize=(12, 5)
# )

# plt.plot(
#     history.history['mae'],
#     label='Train MAE'
# )

# plt.plot(
#     history.history['val_mae'],
#     label='Validation MAE'
# )

# plt.title(
#     'Conv1D Train / Validation MAE'
# )

# plt.xlabel(
#     'Epoch'
# )

# plt.ylabel(
#     'MAE'
# )

# plt.legend()

# plt.grid()

# plt.tight_layout()

# plt.show()


# # ============================================================
# # 32. 마지막 24시간 예측 그래프
# # ============================================================

# plt.figure(
#     figsize=(15, 6)
# )

# plt.plot(
#     range(1, 145),
#     y_cor,
#     label='Actual wd'
# )

# plt.plot(
#     range(1, 145),
#     final_pred_1d,
#     label='Conv1D Prediction'
# )

# plt.title(
#     '24-Hour Wind Direction Prediction'
# )

# plt.xlabel(
#     '10-minute interval'
# )

# plt.ylabel(
#     'Wind Direction (degree)'
# )

# plt.legend()

# plt.grid()

# plt.tight_layout()

# plt.show()


# # ============================================================
# # 33. 결과 저장
# # ============================================================

# result_df = pd.DataFrame({

#     'step': np.arange(
#         1,
#         PREDICT_STEPS + 1
#     ),

#     'actual_wd': y_cor,

#     'predicted_wd': final_pred_1d,

#     'error': (
#         final_pred_1d -
#         y_cor
#     ),

#     'absolute_error': np.abs(
#         final_pred_1d -
#         y_cor
#     )
# })


# print("\n===== 최종 결과 DataFrame =====")

# print(
#     result_df.head(20)
# )


# result_df.to_csv(
#     './_data/kaggle_jena_model/'
#     'jena_conv1d_final_prediction.csv',

#     index=False
# )


# print(
#     "\n결과 저장 완료:"
# )

# print(
#     './_data/kaggle_jena_model/'
#     'jena_conv1d_final_prediction.csv'
# )
