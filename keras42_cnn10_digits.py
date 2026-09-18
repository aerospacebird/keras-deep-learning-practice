
# ============================================================
# Digits Dataset - Conv2D CNN Multi-Class Classification
# ============================================================

import numpy as np
import pandas as pd
import time
import datetime
import os

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


# ============================================================
# 1. DATA 준비
# ============================================================

datasets = load_digits()

x = datasets.data
y = datasets.target

print("x shape :", x.shape)
print("y shape :", y.shape)

# x : (1797, 64)
# y : (1797,)

print("\nClass distribution")
print(np.unique(y, return_counts=True))


# ============================================================
# 2. Train / Test Split
# ============================================================

# One-Hot Encoding 하기 전에
# 원래의 0~9 class label을 이용하여 stratify

x_train, x_test, y_train_raw, y_test_raw = train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=42,
    shuffle=True,
    stratify=y
)

print("\nTrain / Test")
print("x_train :", x_train.shape)
print("x_test  :", x_test.shape)
print("y_train :", y_train_raw.shape)
print("y_test  :", y_test_raw.shape)


# ============================================================
# 3. Pixel Scaling
# ============================================================

# Digits pixel value:
# 0 ~ 16

# CNN에서는 0~1 범위로 정규화

x_train = x_train / 16.0
x_test = x_test / 16.0

print("\nScaling")
print("x_train min :", np.min(x_train))
print("x_train max :", np.max(x_train))
print("x_test min  :", np.min(x_test))
print("x_test max  :", np.max(x_test))


# ============================================================
# 4. 1D -> 2D Image 형태로 변경
# ============================================================

# 기존:
#
# (samples, 64)
#
#        ↓
#
# (samples, 8, 8)
#
#        ↓
#
# (samples, 8, 8, 1)
#
# 마지막 1 = grayscale channel

x_train = x_train.reshape(
    x_train.shape[0],
    8,
    8,
    1
)

x_test = x_test.reshape(
    x_test.shape[0],
    8,
    8,
    1
)

print("\nCNN Input Shape")
print("x_train :", x_train.shape)
print("x_test  :", x_test.shape)


# ============================================================
# 5. One-Hot Encoding
# ============================================================

y_train = to_categorical(
    y_train_raw,
    num_classes=10
)

y_test = to_categorical(
    y_test_raw,
    num_classes=10
)

print("\nOne-Hot Encoding")
print("y_train :", y_train.shape)
print("y_test  :", y_test.shape)


# ============================================================
# 6. CNN 모델 구성
# ============================================================

model = Sequential()


# ------------------------------------------------------------
# CNN Block 1
# ------------------------------------------------------------

# Input:
# (8, 8, 1)

model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        input_shape=(8, 8, 1)
    )
)

# (8, 8, 32)

model.add(BatchNormalization())

model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )
)

# (8, 8, 32)

model.add(MaxPooling2D(pool_size=(2, 2)))

# (4, 4, 32)

model.add(Dropout(0.20))


# ------------------------------------------------------------
# CNN Block 2
# ------------------------------------------------------------

model.add(
    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )
)

# (4, 4, 64)

model.add(BatchNormalization())

model.add(
    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )
)

# (4, 4, 64)

model.add(Dropout(0.20))


# ============================================================
# 7. Feature Extraction -> Classification
# ============================================================

# (4, 4, 64)
#
#        ↓
#
# GlobalAveragePooling2D
#
#        ↓
#
# (64,)

model.add(GlobalAveragePooling2D())

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.20))

model.add(Dense(32, activation='relu'))

# 10 classes
model.add(Dense(10, activation='softmax'))


# ============================================================
# 8. Model Summary
# ============================================================

model.summary()


# ============================================================
# 9. Compile
# ============================================================

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)


# ============================================================
# 10. EarlyStopping
# ============================================================

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=15,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# 11. ModelCheckpoint
# ============================================================

path = './_save/digits_cnn/'

os.makedirs(
    path,
    exist_ok=True
)

date = datetime.datetime.now().strftime(
    "%m%d_%H%M"
)

filename = (
    path
    + 'digits_cnn_'
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
# 12. Training
# ============================================================

start_time = time.time()

hist = model.fit(
    x_train,
    y_train,

    epochs=200,

    # 원래 batch_size=4보다 크게 설정
    batch_size=32,

    validation_split=0.2,

    callbacks=[
        es,
        mcp
    ],

    verbose=1
)

end_time = time.time()


# ============================================================
# 13. Training Time
# ============================================================

print(
    "\nTraining time :",
    round(
        end_time - start_time,
        2
    ),
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
    batch_size=32,
    verbose=1
)

print("\nPrediction shape")
print(y_predict_prob.shape)

# (360, 10)


# ============================================================
# 16. Softmax Probability -> Class
# ============================================================

y_predict = np.argmax(
    y_predict_prob,
    axis=1
)

y_true = np.argmax(
    y_test,
    axis=1
)

print("\nPredicted")
print(y_predict[:30])

print("\nTrue")
print(y_true[:30])


# ============================================================
# 17. Accuracy Score
# ============================================================

acc_score = accuracy_score(
    y_true,
    y_predict
)

print("\n==============================")
print("Accuracy")
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
# 19. Loss Graph
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
plt.title('Digits CNN Loss')

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 20. Accuracy Graph
# ============================================================

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
plt.title('Digits CNN Accuracy')

plt.legend()
plt.grid(True)

plt.show()

