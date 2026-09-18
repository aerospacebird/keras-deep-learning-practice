
# ============================================================
# Covertype Dataset - CNN 7-Class Classification
# ============================================================

import numpy as np
import pandas as pd
import time
import datetime
import os

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


# ============================================================
# 1. DATA 준비
# ============================================================

datasets = fetch_covtype()

x = datasets.data
y = datasets.target

print("x shape :", x.shape)
print("y shape :", y.shape)

# (581012, 54)
# (581012,)

print("\nClass distribution")
print(np.unique(y, return_counts=True))


# ============================================================
# 2. Target 전처리
# ============================================================

# 원래 target : 1 ~ 7
# Keras One-Hot Encoding에서는 0 ~ 6으로 변경

y = y - 1

print("\nTarget after -1")
print(np.unique(y, return_counts=True))


# ============================================================
# 3. Train / Test Split
# ============================================================

x_train, x_test, y_train_raw, y_test_raw = train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=42,
    shuffle=True,
    stratify=y
)

print("\nTrain/Test")
print("x_train :", x_train.shape)
print("x_test  :", x_test.shape)
print("y_train :", y_train_raw.shape)
print("y_test  :", y_test_raw.shape)


# ============================================================
# 4. One-Hot Encoding
# ============================================================

y_train = to_categorical(
    y_train_raw,
    num_classes=7
)

y_test = to_categorical(
    y_test_raw,
    num_classes=7
)

print("\nAfter One-Hot Encoding")
print("y_train :", y_train.shape)
print("y_test  :", y_test.shape)


# ============================================================
# 5. Scaling
# ============================================================

# IMPORTANT:
# scaler는 반드시 train 데이터에만 fit
# test 데이터에는 transform만 수행

scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print("\nScaling")
print("train min :", np.min(x_train))
print("train max :", np.max(x_train))
print("test min  :", np.min(x_test))
print("test max  :", np.max(x_test))


# ============================================================
# 6. CNN 입력 형태로 변경
# ============================================================

# 현재:
# x_train = (464809, 54)
#
# Conv1D 입력:
# (samples, timesteps, channels)
#
# 따라서:
# (464809, 54)
#       ↓
# (464809, 54, 1)

x_train = x_train.reshape(
    x_train.shape[0],
    x_train.shape[1],
    1
)

x_test = x_test.reshape(
    x_test.shape[0],
    x_test.shape[1],
    1
)

print("\nCNN Input Shape")
print("x_train :", x_train.shape)
print("x_test  :", x_test.shape)


# ============================================================
# 7. CNN 모델 구성
# ============================================================

model = Sequential()

# Input
# (54, 1)

model.add(
    Conv1D(
        filters=64,
        kernel_size=3,
        padding='same',
        activation='relu',
        input_shape=(54, 1)
    )
)

# (54, 64)
model.add(BatchNormalization())

model.add(
    Conv1D(
        filters=128,
        kernel_size=3,
        padding='same',
        activation='relu'
    )
)

# (54, 128)
model.add(BatchNormalization())

# (54, 128) -> (27, 128)
model.add(MaxPooling1D(pool_size=2))

model.add(Dropout(0.20))


# ------------------------------------------------------------
# CNN Block 2
# ------------------------------------------------------------

model.add(
    Conv1D(
        filters=128,
        kernel_size=3,
        padding='same',
        activation='relu'
    )
)

# (27, 128)
model.add(BatchNormalization())

model.add(
    Conv1D(
        filters=64,
        kernel_size=3,
        padding='same',
        activation='relu'
    )
)

# (27, 64)
model.add(BatchNormalization())

model.add(Dropout(0.20))


# ============================================================
# 8. Feature Extraction -> Classification
# ============================================================

# (27, 64)
#      ↓
# Global Average Pooling
#      ↓
# (64,)

model.add(GlobalAveragePooling1D())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.30))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.20))

model.add(Dense(32, activation='relu'))

# 7-class classification
model.add(Dense(7, activation='softmax'))


# ============================================================
# 9. Model Summary
# ============================================================

model.summary()


# ============================================================
# 10. Compile
# ============================================================

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)


# ============================================================
# 11. Callback
# ============================================================

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# 12. ModelCheckpoint
# ============================================================

path = './_save/covertype_cnn/'
os.makedirs(path, exist_ok=True)

date = datetime.datetime.now().strftime("%m%d_%H%M")

filename = (
    path
    + 'covertype_cnn_'
    + date
    + '-{epoch:03d}-{val_loss:.4f}.keras'
)

mcp = ModelCheckpoint(
    filepath=filename,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)


# ============================================================
# 13. Training
# ============================================================

start_time = time.time()

hist = model.fit(
    x_train,
    y_train,

    epochs=100,
    batch_size=512,

    validation_split=0.1,

    callbacks=[
        es,
        mcp
    ],

    verbose=1
)

end_time = time.time()


print(
    "\nTraining time :",
    round(end_time - start_time, 2),
    "seconds"
)


# ============================================================
# 14. Evaluation
# ============================================================

result = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print("\n==============================")
print("CNN TEST RESULT")
print("==============================")

print("loss :", result[0])
print("acc  :", round(result[1], 4))


# ============================================================
# 15. Prediction
# ============================================================

y_predict_prob = model.predict(
    x_test,
    batch_size=512,
    verbose=1
)

print("\nPrediction probability shape")
print(y_predict_prob.shape)

# (116203, 7)


# ============================================================
# 16. Softmax -> Class ID
# ============================================================

y_predict = np.argmax(
    y_predict_prob,
    axis=1
)

y_true = np.argmax(
    y_test,
    axis=1
)

print("\nPredicted class")
print(y_predict[:30])

print("\nTrue class")
print(y_true[:30])


# ============================================================
# 17. Accuracy Score
# ============================================================

acc_score = accuracy_score(
    y_true,
    y_predict
)

print("\n==============================")
print("Accuracy Score")
print("==============================")

print("acc_score :", acc_score)


# ============================================================
# 18. Classification Report
# ============================================================

print("\n==============================")
print("Classification Report")
print("==============================")

print(
    classification_report(
        y_true,
        y_predict,
        digits=4
    )
)


# ============================================================
# 19. 학습 시간
# ============================================================

print(
    "\n걸린 시간 :",
    round(end_time - start_time, 2),
    "초"
)


# ============================================================
# 20. Loss / Accuracy 그래프
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.plot(
    hist.history['loss'],
    label='train_loss'
)

plt.plot(
    hist.history['val_loss'],
    label='val_loss'
)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Covertype CNN Loss')

plt.legend()
plt.grid(True)

plt.show()


plt.figure(figsize=(10, 5))

plt.plot(
    hist.history['accuracy'],
    label='train_accuracy'
)

plt.plot(
    hist.history['val_accuracy'],
    label='val_accuracy'
)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Covertype CNN Accuracy')

plt.legend()
plt.grid(True)

plt.show()

