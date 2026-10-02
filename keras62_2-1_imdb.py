
# ============================================================
# 1. 라이브러리
# ============================================================

import numpy as np
import pandas as pd

from tensorflow.keras.datasets import imdb

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LSTM,
    GRU,
    SimpleRNN,
    Embedding
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.metrics import accuracy_score


# ============================================================
# 2. IMDB DATASET
# ============================================================

# 사용할 단어의 최대 개수
#
# 예:
# 1000 → 가장 많이 사용되는 단어 1000개만 사용
#
num_words = 1000


(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=num_words
)


# ============================================================
# 3. DATA 확인
# ============================================================

print()
print("==============================================")
print("IMDB DATASET")
print("==============================================")

print("x_train.shape :", x_train.shape)
print("y_train.shape :", y_train.shape)

print("x_test.shape  :", x_test.shape)
print("y_test.shape  :", y_test.shape)


print()
print("x_train")
print(x_train)

print()
print("y_train")
print(y_train)


# ============================================================
# 4. CLASS 확인
# ============================================================

print()
print("==============================================")
print("CLASS INFORMATION")
print("==============================================")

print("Unique y_train:")
print(np.unique(y_train))

print()

print("Class 개수:")
print(len(np.unique(y_train)))


# IMDB는 감성분류
#
# 0 → 부정(negative)
# 1 → 긍정(positive)
#
num_classes = len(np.unique(y_train))

print()
print("num_classes :", num_classes)


# ============================================================
# 5. DATA TYPE 확인
# ============================================================

print()
print("==============================================")
print("DATA TYPE")
print("==============================================")

print("type(x_train)   :", type(x_train))
print("type(x_train[0]):", type(x_train[0]))

print()

print("첫 번째 리뷰 길이 :", len(x_train[0]))
print("두 번째 리뷰 길이 :", len(x_train[1]))


# ============================================================
# 6. 리뷰 길이 분석
# ============================================================

print()
print("==============================================")
print("REVIEW LENGTH")
print("==============================================")

print(
    "리뷰의 최대길이 :",
    max(len(x) for x in x_train)
)

print(
    "리뷰의 최소길이 :",
    min(len(x) for x in x_train)
)

print(
    "리뷰의 평균길이 :",
    sum(map(len, x_train)) / len(x_train)
)


# ============================================================
# 7. Padding
# ============================================================

# 모든 리뷰를 동일한 길이로 맞춘다.
#
# 200보다 짧으면 앞쪽에 0을 추가
# 200보다 길면 뒤쪽을 잘라낸다.
#
maxlen = 200


x_train = pad_sequences(
    x_train,
    maxlen=maxlen,
    padding='pre',
    truncating='post'
)


x_test = pad_sequences(
    x_test,
    maxlen=maxlen,
    padding='pre',
    truncating='post'
)


print()
print("==============================================")
print("AFTER PADDING")
print("==============================================")

print("x_train.shape :", x_train.shape)
print("x_test.shape  :", x_test.shape)


print()
print("x_train[0]")
print(x_train[0])


# ============================================================
# 8. y 확인
# ============================================================

# IMDB의 y는 이미 0과 1로 구성되어 있으므로
# Reuters처럼 to_categorical()을 사용할 필요가 없다.

print()
print("==============================================")
print("LABEL")
print("==============================================")

print("y_train.shape :", y_train.shape)
print("y_test.shape  :", y_test.shape)

print()

print("y_train[0:20]")
print(y_train[0:20])

print()

print("np.unique(y_train)")
print(np.unique(y_train))


# ============================================================
# 9. GRU MODEL
# ============================================================

model = Sequential()


# ------------------------------------------------------------
# Embedding
# ------------------------------------------------------------
#
# input_dim = 1000
#   → 사용할 단어 사전 크기
#
# output_dim = 46
#   → 단어 하나를 46차원 벡터로 표현
#
# input_length = 200
#   → 하나의 리뷰를 200개 token으로 맞춤
#
# 입력:
#
# (batch_size, 200)
#
# 출력:
#
# (batch_size, 200, 46)
#
# ------------------------------------------------------------

model.add(
    Embedding(
        input_dim=num_words,
        output_dim=46,
        input_length=maxlen
    )
)


# ------------------------------------------------------------
# GRU
# ------------------------------------------------------------
#
# 1000개의 GRU hidden unit
#
# 입력:
# (batch_size, 200, 46)
#
# 출력:
# (batch_size, 1000)
#
# ------------------------------------------------------------

model.add(
    GRU(1000)
)


# ------------------------------------------------------------
# Output Layer
# ------------------------------------------------------------
#
# IMDB는 binary classification
#
# 0 → Negative
# 1 → Positive
#
# sigmoid를 사용하면
# 하나의 확률값을 출력한다.
#
# 예:
#
# 0.12 → Negative
# 0.87 → Positive
#
# ------------------------------------------------------------

model.add(
    Dense(
        1,
        activation='sigmoid'
    )
)


# ============================================================
# 10. MODEL SUMMARY
# ============================================================

print()
print("==============================================")
print("MODEL SUMMARY")
print("==============================================")

model.summary()


# ============================================================
# 11. COMPILE
# ============================================================

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# ============================================================
# 12. CALLBACK
# ============================================================

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)


reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6
)


# ============================================================
# 13. TRAINING
# ============================================================

history = model.fit(
    x_train,
    y_train,

    epochs=50,

    batch_size=32,

    validation_split=0.2,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# 14. MODEL EVALUATION
# ============================================================

loss, acc = model.evaluate(
    x_test,
    y_test,
    verbose=1
)


print()
print("==============================================")
print("TEST RESULT")
print("==============================================")

print("Test Loss     :", loss)
print("Test Accuracy :", acc)


# ============================================================
# 15. PREDICTION
# ============================================================

y_pred = model.predict(
    x_test,
    verbose=0
)


# ============================================================
# 16. 확률 → CLASS
# ============================================================

# sigmoid 결과는 0~1 사이의 확률
#
# 0.5 이상 → 1 Positive
# 0.5 미만 → 0 Negative

y_pred_class = (
    y_pred >= 0.5
).astype(int).reshape(-1)


print()
print("==============================================")
print("PREDICTION")
print("==============================================")

print("y_pred probability:")
print(y_pred[:10])

print()

print("y_pred class:")
print(y_pred_class[:10])

print()

print("y_test:")
print(y_test[:10])


# ============================================================
# 17. Accuracy Score
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred_class
)


print()
print("==============================================")
print("FINAL ACCURACY")
print("==============================================")

print("Accuracy Score :", accuracy)
#############################################################################################

# ==============================================
# FINAL ACCURACY
# ==============================================
# Accuracy Score : 0.81224

#############################################################################################
#model.summary()

# model = Sequential()
# model.add(Embedding())

##################################### SIMPLERNN ##############################################
# Test Loss     : 2.476473808288574
# Test Accuracy : 0.47640249133110046
#####################################SIMPLERNN ################################################
# Test Loss     : 3.922309398651123
# Test Accuracy : 0.4630454182624817
####################################  GRU MODEL ###############################################
# Test Loss     : 1.683680772781372
# Test Accuracy : 0.7524487972259521
####################################  GRU MODEL 500 ###########################################
# Test Loss     : 1.5804482698440552
# Test Accuracy : 0.7528940439224243
#################################### GRU 1000       ###########################################
# Test Loss     : 1.5550652742385864
# Test Accuracy : 0.764915406703949
################################################################################################
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  embedding (Embedding)       (None, 100, 46)           46000     
                                                                 
#  gru (GRU)                   (None, 145)               83955 