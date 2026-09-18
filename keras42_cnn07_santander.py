
# ============================================================
# Kaggle Santander Customer Transaction Prediction
# Conv1D CNN Classification
#
# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data
# ============================================================


# ============================================================
# 0. IMPORT
# ============================================================

import numpy as np
import pandas as pd
import time
import datetime
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)

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

path = './_data/kaggle_santander/'

train_csv = pd.read_csv(
    path + 'train.csv',
    index_col=0
)

test_csv = pd.read_csv(
    path + 'test.csv',
    index_col=0
)

submission = pd.read_csv(
    path + 'sample_submission.csv',
    index_col=0
)


# ------------------------------------------------------------
# Check shape
# ------------------------------------------------------------

print("=" * 70)
print("DATA SHAPE")
print("=" * 70)

print("train :", train_csv.shape)
print("test  :", test_csv.shape)
print("submission :", submission.shape)


# Expected:
#
# train      : (200000, 201)  -> target 포함
# test       : (200000, 200)
# submission : (200000, 1)


print("\nTrain columns")
print(train_csv.columns)

print("\nTrain info")
print(train_csv.info())

print("\nTest info")
print(test_csv.info())


# ============================================================
# 2. MISSING VALUE CHECK
# ============================================================

print("=" * 70)
print("MISSING VALUE")
print("=" * 70)

print("Train missing values:")
print(train_csv.isna().sum().sum())

print("\nTest missing values:")
print(test_csv.isna().sum().sum())


# ============================================================
# 3. X / Y SPLIT
# ============================================================

x = train_csv.drop(
    ['target'],
    axis=1
)

y = train_csv['target']


print("=" * 70)
print("X / Y")
print("=" * 70)

print("x shape :", x.shape)
print("y shape :", y.shape)


# Expected:
#
# x : (200000, 200)
# y : (200000,)


# ============================================================
# 4. TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution")
print(pd.Series(y).value_counts())

print("\nTarget ratio")
print(pd.Series(y).value_counts(normalize=True))

print("\nUnique target")
print(np.unique(y, return_counts=True))


# Expected approximately:
#
# 0 : 179902
# 1 :  20098


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.75,
    shuffle=True,
    random_state=337,
    stratify=y
)


print("=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print("x_train :", x_train.shape)
print("x_test  :", x_test.shape)

print("y_train :", y_train.shape)
print("y_test  :", y_test.shape)


# ============================================================
# 6. SCALING
# ============================================================
#
# IMPORTANT:
#
# scaler must be fitted ONLY on training data.
#
# x_train -> fit_transform
# x_test  -> transform
# test_csv -> transform
#
# ============================================================

scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)

test_csv_scaled = scaler.transform(test_csv)


print("=" * 70)
print("SCALING")
print("=" * 70)

print("x_train min :", np.min(x_train))
print("x_train max :", np.max(x_train))

print("x_test min :", np.min(x_test))
print("x_test max :", np.max(x_test))

print("test min :", np.min(test_csv_scaled))
print("test max :", np.max(test_csv_scaled))


# ============================================================
# 7. CNN INPUT SHAPE
# ============================================================
#
# Original tabular data:
#
# (samples, 200)
#
# Conv1D requires:
#
# (samples, timesteps, channels)
#
# Therefore:
#
# (samples, 200)
#        ↓
# (samples, 200, 1)
#
# 200 = number of features
#   1 = channel
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

test_csv_scaled = test_csv_scaled.reshape(
    test_csv_scaled.shape[0],
    test_csv_scaled.shape[1],
    1
)


print("=" * 70)
print("CNN INPUT SHAPE")
print("=" * 70)

print("x_train :", x_train.shape)
print("x_test  :", x_test.shape)
print("test    :", test_csv_scaled.shape)


# Expected:
#
# x_train : (150000, 200, 1)
# x_test  : (50000, 200, 1)
# test    : (200000, 200, 1)


# ============================================================
# 8. CNN MODEL
# ============================================================

model = Sequential()


# ------------------------------------------------------------
# Conv1D #1
# ------------------------------------------------------------
#
# Input:
# (200, 1)
#
# filters = 64
# kernel_size = 3
#
# Output:
# (200, 64)
#
# padding='same' keeps the feature length = 200
#

model.add(
    Conv1D(
        filters=64,
        kernel_size=3,
        padding='same',
        activation='relu',
        input_shape=(200, 1)
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
        filters=128,
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
        filters=64,
        kernel_size=3,
        padding='same',
        activation='relu'
    )
)


# ------------------------------------------------------------
# Flatten
# ------------------------------------------------------------
#
# (200, 64)
#    ↓
# 12800
#

model.add(
    Flatten()
)


# ------------------------------------------------------------
# Fully Connected Layers
# ------------------------------------------------------------

model.add(
    Dense(
        128,
        activation='relu'
    )
)

model.add(
    Dropout(0.2)
)

model.add(
    Dense(
        64,
        activation='relu'
    )
)

model.add(
    Dense(
        32,
        activation='relu'
    )
)


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------
#
# Binary classification
#
# sigmoid:
# probability between 0 and 1
#

model.add(
    Dense(
        1,
        activation='sigmoid'
    )
)


# ============================================================
# 9. MODEL SUMMARY
# ============================================================

print("=" * 70)
print("MODEL SUMMARY")
print("=" * 70)

model.summary()


# ============================================================
# 10. COMPILE
# ============================================================

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# ============================================================
# 11. EARLY STOPPING
# ============================================================

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# 12. MODEL CHECKPOINT
# ============================================================

date = datetime.datetime.now()

date = date.strftime(
    "%m%d_%H%M"
)

print("date :", date)


# Create directory if it does not exist

save_path = './_save/keras32/'

os.makedirs(
    save_path,
    exist_ok=True
)


filename = (
    save_path
    + 'santander_cnn_'
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
# 13. TRAINING
# ============================================================

start_time = time.time()


hist = model.fit(
    x_train,
    y_train,

    epochs=100,

    batch_size=256,

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
    "Training time :",
    end_time - start_time,
    "seconds"
)


# ============================================================
# 14. EVALUATION
# ============================================================

loss, acc = model.evaluate(
    x_test,
    y_test,
    verbose=0
)


print("=" * 70)
print("TEST RESULT")
print("=" * 70)

print("loss :", loss)
print("acc  :", round(acc, 4))


# ============================================================
# 15. PREDICTION
# ============================================================

y_pred = model.predict(
    x_test,
    verbose=0
)


print("=" * 70)
print("RAW PREDICTION")
print("=" * 70)

print(y_pred[:10])


# ============================================================
# 16. CLASS PREDICTION
# ============================================================
#
# probability
#
# 0.83 -> 1
# 0.21 -> 0
#
# threshold = 0.5
#
# ============================================================

y_pred_class = np.round(
    y_pred
).astype(int)


print("\nPredicted class:")
print(
    y_pred_class[:10].reshape(-1)
)

print("\nActual class:")
print(
    y_test[:10].values
)


# ============================================================
# 17. ACCURACY
# ============================================================

acc_score = accuracy_score(
    y_test,
    y_pred_class
)

print("=" * 70)
print("ACCURACY")
print("=" * 70)

print(
    "accuracy_score :",
    acc_score
)


# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred_class
    )
)


# ============================================================
# 19. ROC-AUC
# ============================================================
#
# Santander has an imbalanced target.
#
# Therefore ROC-AUC is also useful.
#
# IMPORTANT:
# Use probability y_pred,
# NOT rounded y_pred_class.
#
# ============================================================

auc_score = roc_auc_score(
    y_test,
    y_pred
)

print("=" * 70)
print("ROC-AUC")
print("=" * 70)

print(
    "ROC-AUC :",
    auc_score
)


# ============================================================
# 20. SUBMISSION
# ============================================================
#
# IMPORTANT:
#
# For Kaggle submission,
# use sigmoid probability.
#
# DO NOT use np.round().
#
# ============================================================

y_submit = model.predict(
    test_csv_scaled,
    verbose=0
)


print("=" * 70)
print("SUBMISSION PREDICTION")
print("=" * 70)

print("y_submit shape :", y_submit.shape)
print(y_submit[:10])


# ------------------------------------------------------------
# Convert (N,1) -> (N,)
# ------------------------------------------------------------

submission['target'] = y_submit.reshape(-1)


print("\nSubmission")
print(submission.head())

print("\nSubmission shape")
print(submission.shape)


# ============================================================
# 21. SAVE SUBMISSION
# ============================================================

submit_path = path + 'submit/'

os.makedirs(
    submit_path,
    exist_ok=True
)


submit_filename = (
    submit_path
    + 'submit_santander_cnn_'
    + date
    + '.csv'
)


submission.to_csv(
    submit_filename
)


print("=" * 70)
print("SUBMISSION SAVED")
print("=" * 70)

print(submit_filename)


# ============================================================
# 22. HISTORY
# ============================================================

print("=" * 70)
print("HISTORY")
print("=" * 70)

print(hist)

print("\nHistory Dictionary")
print(hist.history)

print("\nLoss")
print(hist.history['loss'])

print("\nValidation Loss")
print(hist.history['val_loss'])


# ============================================================
# 23. LOSS GRAPH
# ============================================================

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


loss_history = hist.history['loss']
val_loss_history = hist.history['val_loss']

epochs = range(
    1,
    len(loss_history) + 1
)


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    epochs,
    loss_history,
    label='Training Loss',
    linewidth=2
)


plt.plot(
    epochs,
    val_loss_history,
    label='Validation Loss',
    linewidth=2
)


plt.title(
    'Kaggle Santander - CNN Loss Curve'
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
# 24. ACCURACY GRAPH
# ============================================================

accuracy_history = hist.history['accuracy']
val_accuracy_history = hist.history['val_accuracy']


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    epochs,
    accuracy_history,
    label='Training Accuracy',
    linewidth=2
)


plt.plot(
    epochs,
    val_accuracy_history,
    label='Validation Accuracy',
    linewidth=2
)


plt.title(
    'Kaggle Santander - CNN Accuracy Curve'
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

