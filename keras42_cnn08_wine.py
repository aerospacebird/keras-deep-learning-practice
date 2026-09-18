
# ============================================================
# Wine Classification
# Conv1D CNN
#
# 목표:
# acc >= 0.95
# ============================================================


# ============================================================
# 0. IMPORT
# ============================================================

import numpy as np
import pandas as pd
import time
import datetime
import os

from sklearn.datasets import load_wine
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

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)


# ============================================================
# 1. DATA
# ============================================================

datasets = load_wine()

print(datasets.DESCR)

print("\nFeature names")
print(datasets.feature_names)


# ------------------------------------------------------------
# X / Y
# ------------------------------------------------------------

x = datasets.data
y = datasets.target


print("\nX shape :", x.shape)
print("Y shape :", y.shape)

# (178, 13)
# (178,)


print("\nTarget")
print(y)


# ============================================================
# 2. CLASS DISTRIBUTION
# ============================================================

print("\nUnique target")
print(
    np.unique(
        y,
        return_counts=True
    )
)

# (array([0, 1, 2]), array([59, 71, 48]))


# ============================================================
# 3. ONE-HOT ENCODING
# ============================================================

from tensorflow.keras.utils import to_categorical

y = to_categorical(y)

print("\nOne-Hot Y")
print(y)

print("\nY shape")
print(y.shape)

# (178, 3)


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=42,
    shuffle=True,
    stratify=y
)


print("=" * 70)
print("TRAIN / TEST")
print("=" * 70)

print("x_train :", x_train.shape)
print("x_test  :", x_test.shape)

print("y_train :", y_train.shape)
print("y_test  :", y_test.shape)

# x_train : (142, 13)
# x_test  : (36, 13)
#
# y_train : (142, 3)
# y_test  : (36, 3)


# ============================================================
# 5. SCALING
# ============================================================
#
# IMPORTANT
#
# Fit scaler ONLY on training data.
#
# Do NOT do:
#
# scaler.fit(x_train)
# scaler.transform(x_train)
# scaler.fit_transform(x_train)
#
# Instead:
#
# x_train = scaler.fit_transform(x_train)
# x_test  = scaler.transform(x_test)
#
# ============================================================

scaler = RobustScaler()

x_train = scaler.fit_transform(
    x_train
)

x_test = scaler.transform(
    x_test
)


print("=" * 70)
print("SCALING")
print("=" * 70)

print(
    "x_train min :",
    np.min(x_train)
)

print(
    "x_train max :",
    np.max(x_train)
)

print(
    "x_test min :",
    np.min(x_test)
)

print(
    "x_test max :",
    np.max(x_test)
)


# ============================================================
# 6. CNN INPUT SHAPE
# ============================================================
#
# Original:
#
# x_train
# (142, 13)
#
# Conv1D requires:
#
# (samples, timesteps, channels)
#
# Therefore:
#
# (142, 13)
#      ↓
# (142, 13, 1)
#
# 13 = Wine features
#  1 = channel
#
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


print("=" * 70)
print("CNN INPUT")
print("=" * 70)

print(
    "x_train :",
    x_train.shape
)

print(
    "x_test  :",
    x_test.shape
)

# x_train : (142, 13, 1)
# x_test  : (36, 13, 1)


# ============================================================
# 7. CNN MODEL
# ============================================================

model = Sequential()


# ------------------------------------------------------------
# Conv1D #1
# ------------------------------------------------------------
#
# Input:
# (13, 1)
#
# filters = 32
# kernel_size = 3
#
# padding='same'
#
# Output:
# (13, 32)
#

model.add(
    Conv1D(
        filters=32,
        kernel_size=3,
        padding='same',
        activation='relu',
        input_shape=(13, 1)
    )
)


# ------------------------------------------------------------
# Batch Normalization
# ------------------------------------------------------------

model.add(
    BatchNormalization()
)


# ------------------------------------------------------------
# Conv1D #2
# ------------------------------------------------------------

model.add(
    Conv1D(
        filters=64,
        kernel_size=3,
        padding='same',
        activation='relu'
    )
)


model.add(
    BatchNormalization()
)


# ------------------------------------------------------------
# Dropout
# ------------------------------------------------------------

model.add(
    Dropout(0.2)
)


# ------------------------------------------------------------
# Conv1D #3
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
# Flatten
# ------------------------------------------------------------
#
# (13, 32)
#    ↓
# 416
#

model.add(
    Flatten()
)


# ============================================================
# 8. FULLY CONNECTED LAYERS
# ============================================================

model.add(
    Dense(
        64,
        activation='relu'
    )
)

model.add(
    Dropout(0.2)
)

model.add(
    Dense(
        32,
        activation='relu'
    )
)

model.add(
    Dense(
        16,
        activation='relu'
    ))


# ============================================================
# 9. OUTPUT
# ============================================================
#
# Wine:
#
# class 0
# class 1
# class 2
#
# Therefore output = 3
#
# softmax:
# probability distribution
#
# probability sum = 1
#
# ============================================================

model.add(
    Dense(
        3,
        activation='softmax'
    )
)


# ============================================================
# 10. MODEL SUMMARY
# ============================================================

print("=" * 70)
print("MODEL SUMMARY")
print("=" * 70)

model.summary()


# ============================================================
# 11. COMPILE
# ============================================================
#
# Multi-class classification
#
# One-hot encoded y
#       ↓
# categorical_crossentropy
#
# ============================================================

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# ============================================================
# 12. EARLY STOPPING
# ============================================================

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# 13. MODEL CHECKPOINT
# ============================================================

date = datetime.datetime.now()

date = date.strftime(
    "%m%d_%H%M"
)

print("date :", date)


save_path = './_save/keras32/'

os.makedirs(
    save_path,
    exist_ok=True
)


filename = (
    save_path
    + 'wine_cnn_'
    + date
    + '-{epoch:04d}-{val_loss:.4f}.keras'
)


mcp = ModelCheckpoint(
    filepath=filename,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)


# ============================================================
# 14. TRAINING
# ============================================================

start_time = time.time()


hist = model.fit(
    x_train,
    y_train,

    epochs=1000,

    batch_size=4,

    validation_split=0.2,

    verbose=1,

    callbacks=[
        es,
        mcp
    ]
)


end_time = time.time()


print("=" * 70)
print("TRAINING TIME")
print("=" * 70)

print(
    "걸린시간 :",
    round(
        end_time - start_time,
        2
    ),
    "초"
)


# ============================================================
# 15. EVALUATION
# ============================================================

result = model.evaluate(
    x_test,
    y_test,
    verbose=0
)


print("=" * 70)
print("TEST RESULT")
print("=" * 70)

print(
    "loss :",
    result[0]
)

print(
    "acc :",
    round(
        result[1],
        4
    )
)


# ============================================================
# 16. PREDICTION
# ============================================================

y_predict = model.predict(
    x_test,
    verbose=0
)


print("=" * 70)
print("RAW PREDICTION")
print("=" * 70)

print(
    y_predict[:10]
)


# ============================================================
# 17. SOFTMAX → CLASS
# ============================================================
#
# Example:
#
# [0.01, 0.97, 0.02]
#
# argmax(axis=1)
#       ↓
# 1
#
# ============================================================

y_predict_class = np.argmax(
    y_predict,
    axis=1
)


# ------------------------------------------------------------
# One-hot y_test → class
# ------------------------------------------------------------

y_test_class = np.argmax(
    y_test,
    axis=1
)


print("\nPredicted class")
print(
    y_predict_class
)

print("\nActual class")
print(
    y_test_class
)


# ============================================================
# 18. ACCURACY SCORE
# ============================================================

acc_score = accuracy_score(
    y_test_class,
    y_predict_class
)


print("=" * 70)
print("ACCURACY")
print("=" * 70)

print(
    "acc_score :",
    acc_score
)


# ============================================================
# 19. CLASSIFICATION REPORT
# ============================================================

print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test_class,
        y_predict_class
    )
)


# ============================================================
# 20. HISTORY
# ============================================================

print("=" * 70)
print("HISTORY")
print("=" * 70)

print(hist)

print("\nHistory Dictionary")
print(hist.history)

print("\nLoss")
print(
    hist.history['loss']
)

print("\nValidation Loss")
print(
    hist.history['val_loss']
)


# ============================================================
# 21. LOSS GRAPH
# ============================================================

import matplotlib.pyplot as plt


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    hist.history['loss'],
    label='Training Loss',
    linewidth=2
)


plt.plot(
    hist.history['val_loss'],
    label='Validation Loss',
    linewidth=2
)


plt.title(
    'Wine CNN - Loss Curve'
)

plt.xlabel(
    'Epoch'
)

plt.ylabel(
    'Loss'
)

plt.legend(
    loc='upper right'
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 22. ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    hist.history['accuracy'],
    label='Training Accuracy',
    linewidth=2
)


plt.plot(
    hist.history['val_accuracy'],
    label='Validation Accuracy',
    linewidth=2
)


plt.title(
    'Wine CNN - Accuracy Curve'
)

plt.xlabel(
    'Epoch'
)

plt.ylabel(
    'Accuracy'
)

plt.legend(
    loc='lower right'
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()

