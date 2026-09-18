
# ============================================================
# Kaggle Bike Sharing Demand
# https://www.kaggle.com/competitions/bike-sharing-demand/data
#
# DNN → CNN(Conv1D) Regression
# ============================================================


# ============================================================
# 0. IMPORT
# ============================================================

import os
import time
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from sklearn.metrics import r2_score, mean_squared_error


# ============================================================
# 1. DATA
# ============================================================

path = './_data/kaggle_bike/'


train_csv = pd.read_csv(
    path + 'train.csv',
    index_col=0
)

test_csv = pd.read_csv(
    path + 'test.csv',
    index_col=0
)

submission = pd.read_csv(
    path + 'sampleSubmission.csv',
    index_col=0
)


print('==================================================')
print('DATA')
print('==================================================')

print('train_csv shape :', train_csv.shape)
print('test_csv shape  :', test_csv.shape)
print('submission shape:', submission.shape)

print(train_csv.columns)

print(train_csv.info())
print(test_csv.info())

print(train_csv.describe())


# ============================================================
# 2. MISSING VALUE CHECK
# ============================================================

print('==================================================')
print('MISSING VALUE')
print('==================================================')

print(train_csv.isna().sum())
print(test_csv.isna().sum())


# ============================================================
# 3. X / Y SPLIT
# ============================================================
#
# casual / registered는 count를 구성하는 값이므로
# 입력 feature에서 제외
#
# X = 8 features
#
# datetime
# season
# holiday
# workingday
# weather
# temp
# atemp
# humidity
# windspeed
#
# Y = count
#

x = train_csv.drop(
    ['casual', 'registered', 'count'],
    axis=1
)

y = train_csv['count']


print('==================================================')
print('X / Y')
print('==================================================')

print('x shape :', x.shape)     # (10886, 8)
print('y shape :', y.shape)     # (10886,)


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.75,
    shuffle=True,
    random_state=500
)


print('==================================================')
print('TRAIN / TEST SPLIT')
print('==================================================')

print('x_train :', x_train.shape)
print('x_test  :', x_test.shape)

print('y_train :', y_train.shape)
print('y_test  :', y_test.shape)


# ============================================================
# 5. TRAIN / VALIDATION SPLIT
# ============================================================

x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    train_size=0.75,
    shuffle=True,
    random_state=500
)


print('==================================================')
print('TRAIN / VALIDATION SPLIT')
print('==================================================')

print('x_train :', x_train.shape)
print('x_val   :', x_val.shape)
print('x_test  :', x_test.shape)


# ============================================================
# 6. SCALING
# ============================================================

scaler = RobustScaler()


# IMPORTANT
# scaler는 training data에만 fit

x_train = scaler.fit_transform(
    x_train
)


# validation / test는 transform만 수행

x_val = scaler.transform(
    x_val
)

x_test = scaler.transform(
    x_test
)


# 실제 Kaggle test data
test_csv_scaled = scaler.transform(
    test_csv
)


print('==================================================')
print('SCALING')
print('==================================================')

print(
    'x_train :',
    np.min(x_train),
    np.max(x_train)
)

print(
    'x_val   :',
    np.min(x_val),
    np.max(x_val)
)

print(
    'x_test  :',
    np.min(x_test),
    np.max(x_test)
)

print(
    'test    :',
    np.min(test_csv_scaled),
    np.max(test_csv_scaled)
)


# ============================================================
# 7. CNN INPUT SHAPE
# ============================================================
#
# DNN input
#
# (samples, 8)
#
# Conv1D input
#
# (samples, steps, channels)
#
# 따라서
#
# (samples, 8)
#       ↓
# (samples, 8, 1)
#
# 8 = feature
# 1 = channel
#


x_train = x_train.reshape(
    -1,
    8,
    1
)

x_val = x_val.reshape(
    -1,
    8,
    1
)

x_test = x_test.reshape(
    -1,
    8,
    1
)

test_csv_scaled = test_csv_scaled.reshape(
    -1,
    8,
    1
)


print('==================================================')
print('CNN INPUT SHAPE')
print('==================================================')

print('x_train :', x_train.shape)
print('x_val   :', x_val.shape)
print('x_test  :', x_test.shape)

print(
    'test    :',
    test_csv_scaled.shape
)


# ============================================================
# 8. CNN MODEL
# ============================================================

model = Sequential()


# ------------------------------------------------------------
# Conv1D Layer 1
# ------------------------------------------------------------

model.add(
    Conv1D(
        filters=32,
        kernel_size=3,
        padding='same',
        activation='relu',
        input_shape=(8, 1)
    )
)


# ------------------------------------------------------------
# Conv1D Layer 2
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
        filters=32,
        kernel_size=3,
        padding='same',
        activation='relu'
    )
)


# ------------------------------------------------------------
# Flatten
# ------------------------------------------------------------

model.add(
    Flatten()
)


# ------------------------------------------------------------
# Dense Layers
# ------------------------------------------------------------

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

model.add(
    Dense(
        16,
        activation='relu'
    )
)

model.add(
    Dense(
        8,
        activation='relu'
    )
)


# ------------------------------------------------------------
# Output Layer
# Regression
# ------------------------------------------------------------

model.add(
    Dense(1)
)


# ============================================================
# 9. MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# 10. COMPILE
# ============================================================

model.compile(
    loss='mse',
    optimizer='adam'
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

save_path = './_save/kaggle_bike_cnn/'

os.makedirs(
    save_path,
    exist_ok=True
)


date = datetime.datetime.now()

print(date)
print(type(date))


date = date.strftime(
    '%m%d_%H%M'
)

print(date)
print(type(date))


filename = (
    save_path
    + 'bike_cnn_'
    + date
    + '-{epoch:04d}-{val_loss:.4f}.keras'
)


print('Model save path:')
print(filename)


mcp = ModelCheckpoint(
    filepath=filename,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)


# ============================================================
# 13. TRAIN
# ============================================================

start_time = time.time()


hist = model.fit(
    x_train,
    y_train,

    epochs=1000,

    batch_size=4,

    verbose=1,

    validation_data=(
        x_val,
        y_val
    ),

    callbacks=[
        es,
        mcp
    ]
)


end_time = time.time()


print('==================================================')
print(
    'Training time :',
    end_time - start_time,
    'seconds'
)
print('==================================================')


# ============================================================
# 14. EVALUATION
# ============================================================

loss = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print('loss :', loss)


# ============================================================
# 15. PREDICTION
# ============================================================

y_predict = model.predict(
    x_test
)


# ============================================================
# 16. R2 SCORE
# ============================================================

r2 = r2_score(
    y_test,
    y_predict
)

print('r2 :', r2)


# ============================================================
# 17. MSE
# ============================================================

mse = mean_squared_error(
    y_test,
    y_predict
)

print('mse :', mse)


# ============================================================
# 18. RMSE
# ============================================================

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_predict
    )
)

print('RMSE :', rmse)


# ============================================================
# 19. KAGGLE TEST PREDICTION
# ============================================================

y_submit = model.predict(
    test_csv_scaled
)


print(
    'y_submit shape :',
    y_submit.shape
)


# ============================================================
# 20. SUBMISSION
# ============================================================

print(submission)


submission['count'] = y_submit.reshape(-1)


print(submission)

print(
    'submission shape :',
    submission.shape
)


# ------------------------------------------------------------
# Submit directory
# ------------------------------------------------------------

submit_path = (
    path
    + 'submit/'
)


os.makedirs(
    submit_path,
    exist_ok=True
)


submission_file = (
    submit_path
    + 'submit_bike_cnn_'
    + date
    + '.csv'
)


submission.to_csv(
    submission_file
)


print('==================================================')
print('Submission saved:')
print(submission_file)
print('==================================================')


# ============================================================
# 21. HISTORY
# ============================================================

print('##########################################################')
print('history')
print('##########################################################')

print(hist)

print('##########################################################')
print('history.history')
print('##########################################################')

print(hist.history)

print('##########################################################')
print('loss')
print('##########################################################')

print(hist.history['loss'])

print('##########################################################')
print('val_loss')
print('##########################################################')

print(hist.history['val_loss'])


# ============================================================
# 22. LOSS GRAPH
# ============================================================

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


plt.figure(
    figsize=(9, 6)
)


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
    'Kaggle Bike Sharing Demand CNN Loss'
)


plt.xlabel(
    'epoch'
)


plt.ylabel(
    'loss'
)


plt.grid()


plt.show()

