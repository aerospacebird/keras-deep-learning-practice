
# ============================================================
# Boston Housing Regression
# DNN → CNN(Conv1D) Version
# ============================================================

import os
import time
import datetime

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


# ============================================================
# 1. DATA
# ============================================================

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print("x_train :", x_train.shape)   # (404, 13)
print("x_test  :", x_test.shape)   # (102, 13)

print("y_train :", y_train.shape)   # (404,)
print("y_test  :", y_test.shape)    # (102,)


# ============================================================
# 2. TRAIN / VALIDATION SPLIT
# ============================================================

x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    train_size=0.75,
    shuffle=True,
    random_state=500
)

print("--------------------------------------------------")
print("After train / validation split")
print("x_train :", x_train.shape)     # (303, 13)
print("x_val   :", x_val.shape)       # (101, 13)
print("x_test  :", x_test.shape)      # (102, 13)
print("--------------------------------------------------")


# ============================================================
# 3. SCALING
# ============================================================

scaler = RobustScaler()

# IMPORTANT:
# scaler는 training data에만 fit
x_train = scaler.fit_transform(x_train)

# validation/test는 transform만 수행
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)


print("Scaled data")
print("x_train min :", np.min(x_train))
print("x_train max :", np.max(x_train))

print("x_val min   :", np.min(x_val))
print("x_val max   :", np.max(x_val))

print("x_test min  :", np.min(x_test))
print("x_test max  :", np.max(x_test))


# ============================================================
# 4. CNN INPUT SHAPE
# ============================================================
#
# Original:
#
# x_train = (303, 13)
#
# Conv1D requires:
#
# (samples, steps, channels)
#
# Therefore:
#
# (303, 13)
#       ↓
# (303, 13, 1)
#
# 13 = number of features
# 1  = channel
#

x_train = x_train.reshape(-1, 13, 1)
x_val   = x_val.reshape(-1, 13, 1)
x_test  = x_test.reshape(-1, 13, 1)


print("--------------------------------------------------")
print("CNN input shape")
print("x_train :", x_train.shape)     # (303, 13, 1)
print("x_val   :", x_val.shape)       # (101, 13, 1)
print("x_test  :", x_test.shape)      # (102, 13, 1)
print("--------------------------------------------------")


# ============================================================
# 5. CNN MODEL
# ============================================================

model = Sequential()


# ------------------------------------------------------------
# Conv1D Layer 1
# ------------------------------------------------------------

model.add(
    Conv1D(
        filters=64,
        kernel_size=3,
        padding='same',
        activation='relu',
        input_shape=(13, 1)
    )
)


# ------------------------------------------------------------
# Conv1D Layer 2
# ------------------------------------------------------------

model.add(
    Conv1D(
        filters=32,
        kernel_size=3,
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
# Conv1D Layer 3
# ------------------------------------------------------------

model.add(
    Conv1D(
        filters=16,
        kernel_size=3,
        padding='same',
        activation='relu'
    )
)


# ------------------------------------------------------------
# Flatten
# ------------------------------------------------------------

model.add(Flatten())


# ------------------------------------------------------------
# Dense Layers
# ------------------------------------------------------------

model.add(Dense(32, activation='relu'))

model.add(Dense(16, activation='relu'))


# ------------------------------------------------------------
# Output Layer
# Boston Housing = Regression
# ------------------------------------------------------------

model.add(Dense(1))


# ============================================================
# 6. MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# 7. COMPILE
# ============================================================

model.compile(
    loss='mse',
    optimizer='adam'
)


# ============================================================
# 8. EARLY STOPPING
# ============================================================

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# 9. MODEL CHECKPOINT
# ============================================================

path = './_save/boston_cnn/'

os.makedirs(path, exist_ok=True)


date = datetime.datetime.now()

print(date)
print(type(date))


date = date.strftime("%m%d_%H%M")

print(date)
print(type(date))


filename = (
    path
    + "boston_cnn_"
    + date
    + "-{epoch:04d}-{val_loss:.4f}.keras"
)


print("Save path :", filename)


mcp = ModelCheckpoint(
    filepath=filename,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)


# ============================================================
# 10. TRAINING
# ============================================================

start_time = time.time()


hist = model.fit(
    x_train,
    y_train,

    epochs=1000,

    batch_size=4,

    verbose=1,

    validation_data=(x_val, y_val),

    callbacks=[es, mcp]
)


end_time = time.time()


print("==================================================")
print("Training time :", end_time - start_time, "seconds")
print("==================================================")


# ============================================================
# 11. EVALUATION
# ============================================================

loss = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print("loss :", loss)


# ============================================================
# 12. HISTORY
# ============================================================

print("##########################################################")
print("history")
print("##########################################################")

print(hist)

print("##########################################################")
print("history.history")
print("##########################################################")

print(hist.history)

print("##########################################################")
print("loss")
print("##########################################################")

print(hist.history['loss'])

print("##########################################################")
print("val_loss")
print("##########################################################")

print(hist.history['val_loss'])


# ============================================================
# 13. LOSS GRAPH
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


plt.legend(loc='upper right')

plt.title('Boston Housing CNN Loss')

plt.xlabel('epoch')

plt.ylabel('loss')

plt.grid()

plt.show()

######################################################## CNN model  @@@@@@@@@@@@@@@@@@@@@@@@
# loss : 25.27750587463379
