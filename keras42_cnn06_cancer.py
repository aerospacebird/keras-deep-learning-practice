
# ============================================================
# Breast Cancer Classification using Conv1D CNN
# ============================================================

import numpy as np
import pandas as pd
import time
import datetime

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# 1. DATA
# ============================================================

datasets = load_breast_cancer()

print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

print("x shape :", x.shape)       # (569, 30)
print("y shape :", y.shape)       # (569,)

print("x type :", type(x))
print("y :", y)

# ------------------------------------------------------------
# Check the number of samples in each class
# ------------------------------------------------------------

print("\n[Class distribution]")
print(pd.Series(y).value_counts())

print("\n[np.unique]")
print(np.unique(y))

print("\n[np.unique + return_counts]")
print(np.unique(y, return_counts=True))


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.75,
    shuffle=True,
    random_state=337,
    stratify=y
)

print("\nTrain/Test shape")
print("x_train :", x_train.shape)     # (426, 30)
print("x_test  :", x_test.shape)      # (143, 30)
print("y_train :", y_train.shape)
print("y_test  :", y_test.shape)


# ============================================================
# 3. SCALING
# ============================================================

scaler = RobustScaler()

# Fit ONLY on training data
x_train = scaler.fit_transform(x_train)

# Transform test data using the same scaler
x_test = scaler.transform(x_test)

print("\nScaled data")
print("train min :", np.min(x_train))
print("train max :", np.max(x_train))

print("test min  :", np.min(x_test))
print("test max  :", np.max(x_test))


# ============================================================
# 4. CNN INPUT SHAPE
# ============================================================
#
# DNN:
# (samples, 30)
#
# Conv1D:
# (samples, 30, 1)
#
# 30 = number of features
# 1  = channel
#
# Example:
# (426, 30)
#       ↓ reshape
# (426, 30, 1)
# ============================================================

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

print("\nCNN input shape")
print("x_train :", x_train.shape)     # (426, 30, 1)
print("x_test  :", x_test.shape)      # (143, 30, 1)


# ============================================================
# 5. MODEL
# ============================================================

model = Sequential()

# Input
# (30, 1)
model.add(
    Conv1D(
        filters=64,
        kernel_size=3,
        padding='same',
        activation='relu',
        input_shape=(30, 1)
    )
)

# (30, 64)
model.add(
    BatchNormalization()
)

model.add(
    Conv1D(
        filters=128,
        kernel_size=3,
        padding='same',
        activation='relu'
    )
)

# (30, 128)
model.add(
    BatchNormalization()
)

model.add(
    Dropout(0.2)
)

model.add(
    Conv1D(
        filters=64,
        kernel_size=3,
        padding='same',
        activation='relu'
    )
)

# (30, 64)

model.add(
    Flatten()
)

# 30 × 64 = 1920
model.add(
    Dense(64, activation='relu')
)

model.add(
    Dropout(0.2)
)

model.add(
    Dense(32, activation='relu')
)

# Binary classification
model.add(
    Dense(1, activation='sigmoid')
)


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# 6. COMPILE
# ============================================================

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# ============================================================
# 7. CALLBACK
# ============================================================

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
    verbose=1
)


# ------------------------------------------------------------
# Create timestamp
# ------------------------------------------------------------

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

print("date :", date)


# ------------------------------------------------------------
# ModelCheckpoint
# ------------------------------------------------------------

path = './_save/keras32/'

filename = (
    path
    + "breast_cancer_cnn_"
    + date
    + "-{epoch:04d}-{val_loss:.4f}.keras"
)

mcp = ModelCheckpoint(
    filepath=filename,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)


# ============================================================
# 8. TRAINING
# ============================================================

start_time = time.time()

hist = model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=4,
    validation_split=0.2,
    verbose=1,
    callbacks=[es, mcp]
)

end_time = time.time()

print("\nTraining time :", end_time - start_time, "seconds")


# ============================================================
# 9. EVALUATION
# ============================================================

loss, acc = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\n==============================")
print("CNN Test Result")
print("==============================")

print("loss :", loss)
print("acc  :", round(acc, 4))


# ============================================================
# 10. PREDICTION
# ============================================================

y_pred = model.predict(
    x_test,
    verbose=0
)

print("\nRaw prediction")
print(y_pred[:10])


# ------------------------------------------------------------
# Convert probability → class
# ------------------------------------------------------------

y_pred_class = np.round(y_pred).astype(int)

print("\nPredicted class")
print(y_pred_class[:10].reshape(-1))

print("\nActual class")
print(y_test[:10])


# ============================================================
# 11. ACCURACY SCORE
# ============================================================

acc_score = accuracy_score(
    y_test,
    y_pred_class
)

print("\naccuracy_score :", acc_score)


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report")
print(
    classification_report(
        y_test,
        y_pred_class
    )
)


# ============================================================
# 13. LOSS / ACCURACY GRAPH
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.plot(
    hist.history['loss'],
    label='loss'
)

plt.plot(
    hist.history['val_loss'],
    label='val_loss'
)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Breast Cancer CNN Loss')

plt.legend()
plt.grid()

plt.show()


# ============================================================
# Accuracy graph
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    hist.history['accuracy'],
    label='accuracy'
)

plt.plot(
    hist.history['val_accuracy'],
    label='val_accuracy'
)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Breast Cancer CNN Accuracy')

plt.legend()
plt.grid()

plt.show()

#####################################################################################################
# loss : 0.38668835163116455
# acc  : 0.965