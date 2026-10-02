
# ============================================================
# 1. 라이브러리
# ============================================================

from tensorflow.keras.datasets import imdb
from tensorflow.keras.datasets import reuters

import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Embedding

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score


# ============================================================
# 2. IMDB 데이터 불러오기
# ============================================================


(x_train, y_train),(x_test, y_test) = imdb.load_data(
        num_words=1000,
        #maxlen = 20,
        #test_split=0.2,
)


print(x_train)
print(x_train.shape, y_train.shape) # (25000,) (25000,)
print(x_test.shape, y_test.shape)   # (25000,) (25000,)
print(y_train) # [ 3  4  3 ... 25  3 25]

# (25000,) (25000,)
# (25000,) (25000,)

print(np.unique(y_train)) #[1 0 0 ... 0 1 0]
                          #[0 1]

print(type(x_train)) #<class 'numpy.ndarray'>
print(type(x_train[0])) # <class 'list'>

print(len(x_train[0]), len(x_train[1])) # 218 189

print("뉴스기사의 최대길이 :", max(len(i) for i in x_train)) # 2494
print("뉴스기사의 최소길이 :", min(len(i) for i in x_train)) # 11
print("뉴스기사의 평균길이 :", sum(map(len, x_train))/len(x_train)) # 238.71364

#exit()

# ============================================================
# 3. 데이터 기본 확인
# ============================================================

print("=" * 70)
print("IMDB DATA")
print("=" * 70)

print("x_train shape :", x_train.shape)
print("y_train shape :", y_train.shape)

print("x_test shape  :", x_test.shape)
print("y_test shape  :", y_test.shape)


# ============================================================
# 4. x_train 확인
# ============================================================

print("\n[ x_train ]")
print(x_train)

print("\n[ x_train 첫 번째 데이터 ]")
print(x_train[0])


# ============================================================
# 5. y_train 확인
# ============================================================

print("\n[ y_train ]")
print(y_train)

print("\n[ y_train unique ]")
print(np.unique(y_train))

# 결과
# [0 1]
#
# 0 = 부정
# 1 = 긍정


# ============================================================
# 6. 데이터 타입 확인
# ============================================================

print("\n[ TYPE ]")

print("type(x_train)   :", type(x_train))
print("type(x_train[0]):", type(x_train[0]))

print("type(y_train)   :", type(y_train))
print("type(y_train[0]):", type(y_train[0]))

# type(x_train)   : <class 'numpy.ndarray'>
# type(x_train[0]): <class 'list'>
# type(y_train)   : <class 'numpy.ndarray'>
# type(y_train[0]): <class 'numpy.int64'>


exit()
# ============================================================
# 7. 리뷰 길이 확인
# ============================================================

print("\n[ REVIEW LENGTH ]")

print(
    "첫 번째 리뷰 길이 :",
    len(x_train[0])
)

print(
    "두 번째 리뷰 길이 :",
    len(x_train[1])
)

print(
    "리뷰 최대 길이 :",
    max(len(i) for i in x_train)
)

print(
    "리뷰 최소 길이 :",
    min(len(i) for i in x_train)
)

print(
    "리뷰 평균 길이 :",
    sum(map(len, x_train)) / len(x_train)
)
# 첫 번째 리뷰 길이 : 218
# 두 번째 리뷰 길이 : 189
# 리뷰 최대 길이 : 2494
# 리뷰 최소 길이 : 11
# 리뷰 평균 길이 : 238.71364
#exit()
# ============================================================
# 8. IMDB 단어 사전 확인
# ============================================================

word_index = imdb.get_word_index() 

print("\n[ WORD INDEX ]")

print("전체 단어 수 :", len(word_index)) #전체 단어 수 : 88584

print("good :", word_index.get("good"))
print("great:", word_index.get("great"))
print("bad  :", word_index.get("bad"))

#exit()
# ============================================================
# 9. Padding
# ============================================================

# 모든 문장의 길이를 동일하게 만든다.
#
# maxlen = 200
#
# 200보다 긴 문장
# → 앞부분을 잘라낸다.
#
# 200보다 짧은 문장
# → 앞부분에 0을 추가한다.
#
# 결과:
#
# (25000, 200)
#
# ============================================================

maxlen = 200

x_train_pad = pad_sequences(
    x_train,
    maxlen=maxlen,
    padding='pre',
    truncating='pre'
)

x_test_pad = pad_sequences(
    x_test,
    maxlen=maxlen,
    padding='pre',
    truncating='pre'
)


print("\n[ PADDING ]")

print("x_train_pad shape :", x_train_pad.shape) #x_train_pad shape : (25000, 200)
print("x_test_pad shape  :", x_test_pad.shape)  #x_test_pad shape  : (25000, 200)

print("\n첫 번째 Padding 결과")
print(x_train_pad[0])

#exit()
# ============================================================
# 10. LSTM MODEL
# ============================================================

model = Sequential()


# ------------------------------------------------------------
# Embedding
# ------------------------------------------------------------
#
# 입력:
#   단어 번호
#
# 출력:
#   64차원 벡터
#
# input_dim = 1000
# → 사용할 단어의 개수
#
# output_dim = 64
# → 각 단어를 64차원으로 표현
#
# input_length = 200
# → 하나의 리뷰 길이
#
# ------------------------------------------------------------

model.add(
    Embedding(
        input_dim= 1000,
        output_dim=64,
        input_length=200,
    )
)


# ------------------------------------------------------------
# LSTM
# ------------------------------------------------------------

model.add(
    LSTM(
        64,
        return_sequences=False
    )
)


# ------------------------------------------------------------
# Dropout
# ------------------------------------------------------------

model.add(
    Dropout(0.3)
)


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------
#
# IMDB는 이진 분류
#
# 0 = 부정
# 1 = 긍정
#
# sigmoid:
# 0 ~ 1 확률
#
# ------------------------------------------------------------

model.add(
    Dense(
        1,
        activation='sigmoid'
    )
)


# ============================================================
# 11. LSTM MODEL SUMMARY
# ============================================================

print("\n")
model.summary()


# ============================================================
# 12. LSTM Compile
# ============================================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# ============================================================
# 13. Callback
# ============================================================

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)


reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6
)


# ============================================================
# 14. LSTM Training
# ============================================================

print("\n")
print("=" * 70)
print("LSTM TRAINING")
print("=" * 70)


history = model.fit(
    x_train_pad,
    y_train,

    epochs=20,

    batch_size=128,

    validation_split=0.2,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    shuffle=True
)


# ============================================================
# 15. LSTM Evaluation
# ============================================================

loss, accuracy = model.evaluate(
    x_test_pad,
    y_test,
    batch_size=128
)


print("\n")
print("=" * 70)
print("LSTM TEST RESULT")
print("=" * 70)

print("LSTM Loss     :", loss)
print("LSTM Accuracy :", accuracy)


# ============================================================
# 16. LSTM Prediction
# ============================================================

probability = model.predict(
    x_test_pad,
    batch_size=128
)


# 확률 → 0 / 1
pred = (
    probability >= 0.5
).astype(int).reshape(-1)


print("\n[ 예측 결과 ]")

print("예측 :", pred[:20])

print("실제 :", y_test[:20])


# ============================================================
# 17. Accuracy 재확인
# ============================================================

accuracy2 = accuracy_score(
    y_test,
    pred
)

print("\nSklearn Accuracy :", accuracy2)


# ============================================================
# 18. GRU MODEL
# ============================================================

gru_model = Sequential()


# Embedding
model.add(
    Embedding(
        input_dim=1000,
        output_dim=64,
        input_length=maxlen
    )
)


# GRU
model.add(
    GRU(
        64,
        return_sequences=False
    )
)


# Dropout
model.add(
    Dropout(0.3)
)


# Output
model.add(
    Dense(
        1,
        activation='sigmoid'
    )
)


# ============================================================
# 19. GRU Summary
# ============================================================

print("\n")
model.summary()


# ============================================================
# 20. GRU Compile
# ============================================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# ============================================================
# 21. GRU Training
# ============================================================

print("\n")
print("=" * 70)
print("GRU TRAINING")
print("=" * 70)


history = model.fit(
    x_train_pad,
    y_train,

    epochs=20,

    batch_size=128,

    validation_split=0.2,

    callbacks=[
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),

        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ],

    shuffle=True
)


# ============================================================
# 22. GRU Evaluation
# ============================================================

loss, accuracy = model.evaluate(
    x_test_pad,
    y_test,
    batch_size=128
)


print("\n")
print("=" * 70)
print("GRU TEST RESULT")
print("=" * 70)

print("GRU Loss     :", loss)
print("GRU Accuracy :", accuracy)


# ============================================================
# 23. GRU Prediction
# ============================================================

probability = model.predict(
    x_test_pad,
    batch_size=128
)


pred = (
    probability >= 0.5
).astype(int).reshape(-1)


accuracy2 = accuracy_score(
    y_test,
    pred
)


print("\nGRU Sklearn Accuracy :", accuracy2)


# ============================================================
# 24. LSTM vs GRU 비교
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "LSTM",
        "GRU"
    ],

    "Loss": [
        loss,
        loss
    ],

    "Accuracy": [
        accuracy,
        accuracy
    ]
})


print("\n")
print("=" * 70)
print("LSTM vs GRU")
print("=" * 70)

print(comparison)


# ============================================================
# 25. 새로운 문장 처리 함수
# ============================================================

def encode_review(
    text,
    word_index,
    num_words=1000
):

    words = text.lower().split()

    sequence = []

    for word in words:

        # 간단한 구두점 제거
        word = (
            word
            .replace(".", "")
            .replace(",", "")
            .replace("!", "")
            .replace("?", "")
            .replace(":", "")
            .replace(";", "")
        )

        # IMDB의 특수 token 처리
        #
        # 0 : padding
        # 1 : start
        # 2 : OOV
        #
        # 실제 단어는 index + 3
        #

        index = word_index.get(word, -1)

        if index == -1:

            # 모르는 단어
            sequence.append(2)

        elif index + 3 < num_words:

            sequence.append(index + 3)

        else:

            # 1000번째 이후 단어
            sequence.append(2)


    return sequence


# ============================================================
# 26. 새로운 문장 LSTM 추론
# ============================================================

def predict_lstm(text):

    sequence = encode_review(
        text,
        word_index,
        num_words=1000
    )

    sequence = pad_sequences(
        [sequence],
        maxlen=maxlen,
        padding='pre',
        truncating='pre'
    )

    probability = model.predict(
        sequence,
        verbose=0
    )[0][0]


    if probability >= 0.5:

        sentiment = "긍정"

    else:

        sentiment = "부정"


    print("\n")
    print("=" * 70)
    print("LSTM SENTIMENT PREDICTION")
    print("=" * 70)

    print("문장       :", text)
    print("긍정 확률  :", probability)
    print("부정 확률  :", 1 - probability)
    print("판정       :", sentiment)

    return probability


# ============================================================
# 27. 새로운 문장 GRU 추론
# ============================================================

def predict_gru(text):

    sequence = encode_review(
        text,
        word_index,
        num_words=1000
    )

    sequence = pad_sequences(
        [sequence],
        maxlen=maxlen,
        padding='pre',
        truncating='pre'
    )

    probability = gru_model.predict(
        sequence,
        verbose=0
    )[0][0]


    if probability >= 0.5:

        sentiment = "긍정"

    else:

        sentiment = "부정"


    print("\n")
    print("=" * 70)
    print("GRU SENTIMENT PREDICTION")
    print("=" * 70)

    print("문장       :", text)
    print("긍정 확률  :", probability)
    print("부정 확률  :", 1 - probability)
    print("판정       :", sentiment)

    return probability


# ============================================================
# 28. 새로운 영화 리뷰 테스트
# ============================================================

predict_lstm(
    "This movie was absolutely fantastic and amazing"
)

predict_lstm(
    "This movie was terrible and boring"
)

predict_lstm(
    "I really loved this movie"
)

predict_lstm(
    "I hated this movie"
)

# ============================================================
# 29. GRU 테스트
# ============================================================

predict_gru(
    "This movie was absolutely fantastic and amazing"
)

predict_gru(
    "This movie was terrible and boring"
)

predict_gru(
    "I really loved this movie"
)

predict_gru(
    "I hated this movie"
)

###################################################################################
#======================================================================
# GRU TEST RESULT
# ======================================================================
# GRU Loss     : 0.3242255747318268
# GRU Accuracy : 0.8650000095367432
# 196/196 [==============================] - 1s 4ms/step

# GRU Sklearn Accuracy : 0.865
######################################################################################
# ======================================================================
# GRU TEST RESULT
# ======================================================================
# GRU Loss     : 0.6931636929512024
# GRU Accuracy : 0.5
# 196/196 [==============================] - 1s 5ms/step

# GRU Sklearn Accuracy : 0.5