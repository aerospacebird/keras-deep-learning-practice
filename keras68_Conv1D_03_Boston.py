
# ============================================================
# Boston Housing
# CNN Conv2D Regression
# ============================================================

from sklearn.datasets import fetch_california_housing, load_diabetes

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
)

from tensorflow.keras.datasets import boston_housing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    MaxAbsScaler,
    RobustScaler
)

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


# ============================================================
# 1. 데이터
# ============================================================

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print("Original data")
print("x_train :", x_train.shape)     # (404, 13)
print("x_test  :", x_test.shape)      # (102, 13)
print("y_train :", y_train.shape)     # (404,)
print("y_test  :", y_test.shape)      # (102,)


# ============================================================
# 2. Scaling
# ============================================================

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()

scaler = RobustScaler()


# Train 데이터로 scaler 학습
x_train = scaler.fit_transform(x_train)

# Test 데이터는 transform만 수행
x_test = scaler.transform(x_test)


print()
print("Scaling 결과")

print("x_train min :", np.min(x_train))
print("x_train max :", np.max(x_train))

print("x_test min :", np.min(x_test))
print("x_test max :", np.max(x_test))


# ============================================================
# 3. Train / Validation 분리
# ============================================================

x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    train_size=0.75,
    shuffle=True,
    random_state=500
)

print()
print("Train / Validation / Test")

print("x_train :", x_train.shape)
print("y_train :", y_train.shape)

print("x_val   :", x_val.shape)
print("y_val   :", y_val.shape)

print("x_test  :", x_test.shape)
print("y_test  :", y_test.shape)


# ============================================================
# 4. Conv2D 입력 형태로 변환
# ============================================================

# 현재
#
# x_train : (303, 13)
# x_val   : (101, 13)
# x_test  : (102, 13)
#
# Conv2D 입력
#
# (samples, height, width, channels)
#
# 따라서
#
# (303, 13)
#       ↓
# (303, 13, 1, 1)


x_train = x_train.reshape(
    -1,
    13,
    1,
    1
)

x_val = x_val.reshape(
    -1,
    13,
    1,
    1
)

x_test = x_test.reshape(
    -1,
    13,
    1,
    1
)


print()
print("Conv2D Input Shape")

print("x_train :", x_train.shape)
print("x_val   :", x_val.shape)
print("x_test  :", x_test.shape)
# x_train : (303, 13, 1, 1)
# x_val   : (101, 13, 1, 1)
# x_test  : (102, 13, 1, 1)

x_train = x_train.reshape(-1, 13, 1)
x_val   = x_val.reshape(-1, 13, 1)
x_test  = x_test.reshape(-1, 13, 1)

print("x_train :", x_train.shape)
print("x_val   :", x_val.shape)
print("x_test  :", x_test.shape)

# x_train : (303, 13, 1)
# x_val   : (101, 13, 1)
# x_test  : (102, 13, 1)

#exit()
# ============================================================
# 5. CNN Conv1D 모델 구성
# ============================================================
model = Sequential()
# Input
# (13, 1)

model.add(
    Conv1D(
        filters=64,
        kernel_size=3,
        activation='relu',
        input_shape=(13, 1)
    )
)

# Output
# (11, 64)


model.add(
    Conv1D(
        filters=32,
        kernel_size=3,
        activation='relu'
    )
)

# Output
# (9, 32)


model.add(
    Conv1D(
        filters=32,
        kernel_size=2,
        activation='relu'
    )
)

# Output
# (8, 32)


model.add(
    Dropout(0.2)
)


model.add(
    Conv1D(
        filters=16,
        kernel_size=2,
        activation='relu'
    )
)

# Output
# (7, 16)


model.add(
    Dropout(0.2)
)


model.add(
    Conv1D(
        filters=32,
        kernel_size=2,
        activation='relu'
    )
)

# Output
# (6, 32)


# ============================================================
# 6. Global Average Pooling
# ============================================================

model.add(
    GlobalAveragePooling1D()
)

# (6, 32)
#      ↓
# (32)


# ============================================================
# 7. Dense Layer
# ============================================================

model.add(
    Dense(
        32,
        activation='relu'
    )
)

model.add(
    Dropout(0.2)
)

model.add(
    Dense(
        16,
        activation='relu'
    )
)


# Regression Output
model.add(
    Dense(1)
)


# ============================================================
# 8. Model Summary
# ============================================================

model.summary()
# ============================================================
# 5. CNN Conv2D 모델 구성
# ============================================================

# model = Sequential()


# # Input
# # (13, 1, 1)

# model.add(
#     Conv2D(
#         filters=64,
#         kernel_size=(3, 1),
#         activation='relu',
#         input_shape=(13, 1, 1)
#     )
# )

# # Output
# # (11, 1, 64)


# model.add(
#     Conv2D(
#         filters=32,
#         kernel_size=(3, 1),
#         activation='relu'
#     )
# )

# # Output
# # (9, 1, 32)


# model.add(
#     Conv2D(
#         filters=32,
#         kernel_size=(2, 1),
#         activation='relu'
#     )
# )

# # Output
# # (8, 1, 32)


# model.add(
#     Dropout(0.2)
# )


# model.add(
#     Conv2D(
#         filters=16,
#         kernel_size=(2, 1),
#         activation='relu'
#     )
# )

# # Output
# # (7, 1, 16)


# model.add(
#     Dropout(0.2)
# )


# model.add(
#     Conv2D(
#         filters=32,
#         kernel_size=(2, 1),
#         activation='relu'
#     )
# )

# # Output
# # (6, 1, 32)


# # ============================================================
# # 6. Global Average Pooling
# # ============================================================

# model.add(
#     GlobalAveragePooling2D()
# )

# # (6, 1, 32)
# #       ↓
# # (32)


# # ============================================================
# # 7. Dense Layer
# # ============================================================

# model.add(
#     Dense(
#         32,
#         activation='relu'
#     )
# )

# model.add(
#     Dropout(0.2)
# )

# model.add(
#     Dense(
#         16,
#         activation='relu'
#     )
# )


# # Regression Output
# model.add(
#     Dense(1)
# )


# # ============================================================
# # 8. Model Summary
# # ============================================================

# model.summary()


# ============================================================
# 9. Compile
# ============================================================

model.compile(
    loss='mse',
    optimizer='adam'
)


# ============================================================
# 10. EarlyStopping
# ============================================================

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# 11. Training
# ============================================================

start_time = time.time()

hist = model.fit(
    x_train,
    y_train,

    epochs=500,

    batch_size=32,

    verbose=1,

    validation_data=(
        x_val,
        y_val
    ),

    callbacks=[es]
)

end_time = time.time()


print()
print("=======================================")
print("Training time :", end_time - start_time)
print("=======================================")


# ============================================================
# 12. 평가
# ============================================================

loss = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print()
print("loss :", loss)


# ============================================================
# 13. 예측
# ============================================================

y_predict = model.predict(
    x_test,
    verbose=0
)


print()
print("Actual")
print(y_test[:10])

print()
print("Prediction")
print(y_predict[:10].flatten())


# ============================================================
# 14. R2 Score
# ============================================================

from sklearn.metrics import (
    r2_score,
    mean_squared_error
)

r2 = r2_score(
    y_test,
    y_predict
)

print()
print("R2 :", r2)


# ============================================================
# 15. MSE
# ============================================================

mse = mean_squared_error(
    y_test,
    y_predict
)

print()
print("MSE :", mse)


# ============================================================
# 16. RMSE
# ============================================================

def RMSE(
    y_test,
    y_predict
):

    return np.sqrt(
        mean_squared_error(
            y_test,
            y_predict
        )
    )


rmse = RMSE(
    y_test,
    y_predict
)

print()
print("RMSE :", rmse)


# ============================================================
# 17. History
# ============================================================

print()
print("################################################")
print("history")
print("################################################")

print(hist)

print()
print(hist.history)


print()
print("################################################")
print("loss")
print("################################################")

print(hist.history['loss'])


print()
print("################################################")
print("val_loss")
print("################################################")

print(hist.history['val_loss'])


print()
print("Epochs :", len(hist.history['loss']))


# ============================================================
# 18. Loss Graph
# ============================================================

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(9, 6))

plt.plot(
    hist.history['loss'][3:],
    label='loss'
)

plt.plot(
    hist.history['val_loss'][3:],
    label='val_loss'
)

plt.legend(
    loc='upper right'
)

plt.title(
    'Boston Housing CNN Conv2D Loss'
)

plt.xlabel(
    'epoch'
)

plt.ylabel(
    'loss'
)

plt.grid()

plt.show()

################################################## RESULT CNN CONV2D    ################################################
# R2 : 0.6294630297176224

# MSE : 30.84492342517191

# RMSE : 5.553820615141608
###################################################  RESULT CNN CONV1D   ###############################################
# Epoch 271: early stopping

# =======================================
# Training time : 15.832226753234863
# =======================================
# 4/4 [==============================] - 0s 37ms/step - loss: 22.2558

# loss : 22.255821228027344

# Actual
# [ 7.2 18.8 19.  27.  22.2 24.5 31.2 22.9 20.5 23.2]

# Prediction
# [ 8.7969475 17.85697   20.770727  22.69381   22.136198  18.399906
#  29.187246  22.424105  18.802843  18.5338   ]

# R2 : 0.7326430297606898

# MSE : 22.25582313131384

# RMSE : 4.717607776332602
