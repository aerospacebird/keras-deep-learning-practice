import numpy as np
import time

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split


# ============================================================
# 1. DATA
# ============================================================

docs = [
    '너무 재미있다',
    '참 최고예요',
    '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다',
    '한 번 더 보고 싶어요',
    '글쎄',
    '별로에요',
    '생각보다 지루해요',
    '연기가 어색해요',
    '재미없어요',
    '너무 재미없다',
    '참 재밋네요',
    '개똥이 바보',
    '말똥이 잘생겼다',
    '길동이 또 구라친다'
]


labels = np.array([
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 0,
    0, 1, 0, 1, 0
])


# ============================================================
# 2. TOKENIZER
# ============================================================

token = Tokenizer()

token.fit_on_texts(docs)

print("\n==============================")
print("WORD INDEX")
print("==============================")

print(token.word_index)


vocab_size = len(token.word_index) + 1

print("\nVocab Size :", vocab_size)    # Vocab Size : 31


# ============================================================
# 3. TEXT → INTEGER SEQUENCE
# ============================================================

x = token.texts_to_sequences(docs)

print("\n==============================")
print("INTEGER SEQUENCE")
print("==============================")

print(x)
# ==============================
# INTEGER SEQUENCE
# ==============================
# [[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23], [24, 25], [26, 27], [28, 29, 30]]

#exit()
# ============================================================
# 4. PADDING
# ============================================================

maxlen = 5

x = pad_sequences(
    x,
    padding='pre',
    maxlen=maxlen,
    truncating='post'
)


print("\n==============================")
print("PADDED X")
print("==============================")

print(x)

# ==============================
# PADDED X
# ==============================
# [[ 0  0  0  2  3]
#  [ 0  0  0  1  4]
#  [ 0  0  1  5  6]
#  [ 0  0  7  8  9]
#  [10 11 12 13 14]
#  [ 0  0  0  0 15]
#  [ 0  0  0  0 16]
#  [ 0  0  0 17 18]
#  [ 0  0  0 19 20]
#  [ 0  0  0  0 21]
#  [ 0  0  0  2 22]
#  [ 0  0  0  1 23]
#  [ 0  0  0 24 25]
#  [ 0  0  0 26 27]
#  [ 0  0 28 29 30]]

print("\nX Shape :", x.shape) # X Shape : (15, 5)

#exit()
# ============================================================
# 5. ONE-HOT ENCODING
# ============================================================
#
# 현재 x:
#
# (15, 5)
#
# 각 정수 token을 One-Hot Vector로 변환
#
# vocab_size = 31
#
# 결과:
#
# (15, 5, 31)
#
# ============================================================

x = to_categorical(
    x,
    num_classes=vocab_size  # 즉 LSTM 입장에서 한 timestep의 feature가 1개가 아니라 31개가 됩니다.
)


print("\n==============================")
print("ONE-HOT ENCODING")
print("==============================")

print("X Shape :", x.shape)  # X Shape : (15, 5, 31)

print("\n첫 번째 문장의 One-Hot:")
print(x[0])

print("\n첫 번째 문장 Shape:")
print(x[0].shape)  #X Shape : (15, 5)

#exit()
# 첫 번째 문장의 One-Hot:
# [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#   0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#   0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#   0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#   0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#   0. 0. 0. 0. 0. 0. 0.]]

#exit()
# ============================================================
# 6. DATA SPLIT
# ============================================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    labels,
    train_size=0.75,
    random_state=43,
    stratify=labels
)


print("\n==============================")
print("DATA SPLIT")
print("==============================")

print("x_train.shape :", x_train.shape)
print("x_test.shape  :", x_test.shape)
print("y_train.shape :", y_train.shape)
print("y_test.shape  :", y_test.shape)

# ==============================
# DATA SPLIT
# ==============================
# x_train.shape : (11, 5, 31)
# x_test.shape  : (4, 5, 31)
# y_train.shape : (11,)
# y_test.shape  : (4,)

#exit()
# ============================================================
# 7. LSTM MODEL
# ============================================================
#
# input_shape:
#
# (timesteps, features)
#
# timesteps = 5
# features  = vocab_size = 31
#
# 따라서:
#
# input_shape = (5, 31)
#
# ============================================================

model = Sequential()


model.add(
    LSTM(
        64,
        activation='tanh',
        input_shape=(maxlen, vocab_size)
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
    Dropout(0.2)
)


model.add(
    Dense(
        16,
        activation='relu'
    )
)


model.add(
    Dense(
        1,
        activation='sigmoid'
    )
)


# ============================================================
# 8. COMPILE
# ============================================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# ============================================================
# 9. MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# 10. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)


# ============================================================
# 11. TRAINING
# ============================================================

start = time.time()


history = model.fit(
    x_train,
    y_train,
    epochs=200,
    batch_size=4,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)


end = time.time()


print("\n==============================")
print("TRAINING TIME")
print("==============================")

print("Training time :", end - start)


# ============================================================
# 12. EVALUATION
# ============================================================

loss, accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)


print("\n==============================")
print("TEST RESULT")
print("==============================")

print("Test Loss     :", loss)
print("Test Accuracy :", accuracy)


# ============================================================
# 13. TEST PREDICTION
# ============================================================

y_pred = model.predict(
    x_test,
    verbose=0
)


y_pred_class = (
    y_pred > 0.5
).astype(int)


print("\n==============================")
print("TEST PREDICTION")
print("==============================")


print("\ny_test:")
print(y_test)


print("\ny_pred probability:")
print(y_pred)


print("\ny_pred class:")
print(y_pred_class)


# ============================================================
# 14. NEW SENTENCE
# ============================================================

new_sentence = [
    '개똥이 잘생겼다'
]


# ============================================================
# 15. TEXT → SEQUENCE
# ============================================================

new_x = token.texts_to_sequences(
    new_sentence
)


print("\n==============================")
print("NEW SENTENCE")
print("==============================")


print("\n1. Original:")
print(new_sentence)
# 1. Original:
# ['개똥이 잘생겼다']

print("\n2. Sequence:")
print(new_x)
# 2. Sequence:
# [[24, 27]]

# ============================================================
# 16. PADDING
# ============================================================

new_x = pad_sequences(
    new_x,
    padding='pre',
    maxlen=maxlen,
    truncating='post'
)


print("\n3. Padding:")
print(new_x)

print("Shape:", new_x.shape)
# 3. Padding:
# [[ 0  0  0 24 27]]
# Shape: (1, 5)

# ============================================================
# 17. ONE-HOT ENCODING
# ============================================================

new_x = to_categorical(
    new_x,
    num_classes=vocab_size
)


print("\n4. One-Hot Shape:")
print(new_x.shape)

# 4. One-Hot Shape:
# (1, 5, 31)  # 여기서 31은 다음 의미입니다.
# 0  = padding
# 1  = 참
# 2  = 너무
# 3  = 재미있다
# ...
# 24 = 개똥이
# 27 = 잘생겼다
# ...
# 30 = 구라친다





print("\nOne-Hot Vector:")
print(new_x)

#One-Hot Vector:
# [[[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#    0. 0. 0. 0. 0. 0. 0. 0.]
#   [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#    0. 0. 0. 0. 0. 0. 0. 0.]
#   [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#    0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#    0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#    0. 0. 0. 0. 1. 0. 0. 0.]]]

# ============================================================
# 18. PREDICTION
# ============================================================

new_pred = model.predict(
    new_x,
    verbose=0
)


print("\n5. Prediction Probability:")
print(new_pred)

# 5. Prediction Probability:
# [[0.47674155]]

# ============================================================
# 19. CLASSIFICATION
# ============================================================

new_pred_class = (
    new_pred > 0.5
).astype(int)


print("\n6. Prediction Class:")
print(new_pred_class)
# 6. Prediction Class:
# [[0]]

# ============================================================
# 20. FINAL RESULT
# ============================================================

if new_pred_class[0][0] == 1:

    print("\n================================")
    print("[긍정]")
    print(new_sentence[0])
    print("================================")

else:

    print("\n================================")
    print("[부정]")
    print(new_sentence[0])
    print("================================")

# ================================
# [부정]
# 개똥이 잘생겼다
# ================================

#############################################################################################################


# import numpy as np
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# import time
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.layers import Dense,SimpleRNN,LSTM,GRU 

# #1. 데이터
# docs = ['너무 재미있다', '참 최고예요', '참 잘만든 영화에요',
#         '추천하고 싶은 영화입니다', '한 번 더 보고 싶어요', '글쎄',
#         '별로에요', '생각보다 지루해요', '연기가 어색해요',
#         '재미없어요', '너무 재미없다', '참 재밋네요',
#         '개똥이 바보', '말똥이 잘생겼다', '길동이 또 구라친다',
#         ]

# labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])  # y

# token = Tokenizer()
# token.fit_on_texts(docs)
# print(token.word_index)
# # {'참': 1, '너무': 2, '재미있다': 3, '최고예요': 4, '잘만든': 5, '영화에요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10, '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밋네요': 23, '개똥이': 24, '바보': 25, '말똥이': 26, '잘생겼다': 27, '길동이': 28, '또': 29, '구라친다': 30}
# x = token.texts_to_sequences(docs)
# print(x)
# # [[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23], [24, 25], [26, 27], [28, 29, 30]]
# ################################ 패딩 ################################################
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# padded_x = pad_sequences(x, 
#                          padding='pre',  #'post'
#                          maxlen = 5,
#                          truncating='post'
#                          ) #'post'

# print(padded_x)
# print(padded_x.shape) # (15, 5)

# x = padded_x
# y = labels

# # [[ 0  0  0  2  3]
# #  [ 0  0  0  1  4]
# #  [ 0  0  1  5  6]
# #  [ 0  0  7  8  9]
# #  [10 11 12 13 14]
# #  [ 0  0  0  0 15]
# #  [ 0  0  0  0 16]
# #  [ 0  0  0 17 18]
# #  [ 0  0  0 19 20]
# #  [ 0  0  0  0 21]
# #  [ 0  0  0  2 22]
# #  [ 0  0  0  1 23]
# #  [ 0  0  0 24 25]
# #  [ 0  0  0 26 27]
# #  [ 0  0 28 29 30]]
# # (15, 5)
# ######################################### 데이터 분리 ###################################

# x_train, x_test, y_train, y_test = train_test_split(
#                                                    x, y,train_size= 0.75,
#                                                    random_state=43,
# )

# print(x.shape, y.shape) # (15, 5) (15,)


# x = x.reshape(15, 5, 1)

# print(x.shape)    # (15, 5, 1)
# #exit()




# #2.  DNN MODEL 구성하시오?

# # ============================================================
# # 5. DATA SPLIT
# # ============================================================

# x_train, x_test, y_train, y_test = train_test_split(
#     x,
#     y,
#     train_size=0.75,
#     random_state=43,
#     stratify=y
# )

# print("\n==============================")
# print("x_train.shape :", x_train.shape)
# print("x_test.shape  :", x_test.shape)
# print("y_train.shape :", y_train.shape)
# print("y_test.shape  :", y_test.shape)
# print("==============================")
# # ==============================
# # x_train.shape : (11, 5)
# # x_test.shape  : (4, 5)
# # y_train.shape : (11,)
# # y_test.shape  : (4,)
# # ==============================

# # import numpy as np

# # from tensorflow.keras.preprocessing.text import Tokenizer
# # from tensorflow.keras.preprocessing.sequence import pad_sequences

# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import Dense, Dropout, LSTM

# # import time

# # from sklearn.model_selection import train_test_split

# # from tensorflow.keras.callbacks import EarlyStopping


# # # ============================================================
# # # 1. DATA
# # # ============================================================

# # docs = [
# #     '너무 재미있다',
# #     '참 최고예요',
# #     '참 잘만든 영화에요',
# #     '추천하고 싶은 영화입니다',
# #     '한 번 더 보고 싶어요',
# #     '글쎄',
# #     '별로에요',
# #     '생각보다 지루해요',
# #     '연기가 어색해요',
# #     '재미없어요',
# #     '너무 재미없다',
# #     '참 재밋네요',
# #     '개똥이 바보',
# #     '말똥이 잘생겼다',
# #     '길동이 또 구라친다',
# # ]

# # labels = np.array([
# #     1, 1, 1, 1, 1,
# #     0, 0, 0, 0, 0,
# #     0, 1, 0, 1, 0
# # ])


# # # ============================================================
# # # 2. TOKENIZER
# # # ============================================================

# # token = Tokenizer()

# # token.fit_on_texts(docs)

# # print("\nWord Index:")
# # print(token.word_index)


# # # ============================================================
# # # 3. TEXT → SEQUENCE
# # # ============================================================

# # x = token.texts_to_sequences(docs)

# # print("\nSequence:")
# # print(x)


# # # ============================================================
# # # 4. PADDING
# # # ============================================================

# # padded_x = pad_sequences(
# #     x,
# #     padding='pre',
# #     maxlen=5,
# #     truncating='post'
# # )

# # print("\nPadded X:")
# # print(padded_x)

# # print("\nPadded X Shape:")
# # print(padded_x.shape)
# # # (15, 5)


# # # ============================================================
# # # 5. LSTM INPUT SHAPE
# # # ============================================================

# # x = padded_x
# # y = labels


# # # ------------------------------------------------------------
# # # DNN
# # # (samples, features)
# # #
# # # LSTM
# # # (samples, timesteps, features)
# # # ------------------------------------------------------------

# # x = x.reshape(
# #     x.shape[0],
# #     x.shape[1],
# #     1
# # )

# # print("\nLSTM Input Shape:")
# # print(x.shape)
# # # (15, 5, 1)


# # # ============================================================
# # # 6. TRAIN / TEST SPLIT
# # # ============================================================

# # x_train, x_test, y_train, y_test = train_test_split(
# #     x,
# #     y,
# #     train_size=0.75,
# #     random_state=43
# # )

# # print("\nx_train shape :", x_train.shape)
# # print("x_test shape  :", x_test.shape)
# # print("y_train shape :", y_train.shape)
# # print("y_test shape  :", y_test.shape)


# # ============================================================
# # 7. LSTM MODEL
# # ============================================================

# model = Sequential()


# # ------------------------------------------------------------
# # LSTM
# #
# # input_shape = (timesteps, features)
# #
# # timesteps = 5
# # features  = 1
# # ------------------------------------------------------------

# model.add(
#     LSTM(
#         64,
#         activation='tanh',
#         input_shape=(5, 1)
#     )
# )


# model.add(
#     Dropout(0.2)
# )


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


# model.add(
#     Dense(
#         1,
#         activation='sigmoid'
#     )
# )


# # ============================================================
# # 8. MODEL COMPILE
# # ============================================================

# model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )


# # ============================================================
# # 9. MODEL SUMMARY
# # ============================================================

# model.summary()


# # ============================================================
# # 10. EARLY STOPPING
# # ============================================================

# early_stopping = EarlyStopping(
#     monitor='val_loss',
#     patience=20,
#     restore_best_weights=True
# )


# # ============================================================
# # 11. MODEL TRAINING
# # ============================================================

# start = time.time()


# history = model.fit(
#     x_train,
#     y_train,
#     epochs=200,
#     batch_size=4,
#     validation_split=0.2,
#     callbacks=[early_stopping],
#     verbose=1
# )


# end = time.time()


# print("\nTraining time :", end - start)


# # ============================================================
# # 12. EVALUATION
# # ============================================================

# loss, accuracy = model.evaluate(
#     x_test,
#     y_test,
#     verbose=0
# )


# print("\n==============================")
# print("Test Loss     :", loss)
# print("Test Accuracy :", accuracy)
# print("==============================")


# # ============================================================
# # 13. PREDICTION
# # ============================================================

# y_pred = model.predict(x_test)


# y_pred_class = (
#     y_pred > 0.5
# ).astype(int)


# print("\ny_test:")
# print(y_test)


# print("\ny_pred probability:")
# print(y_pred)


# print("\ny_pred class:")
# print(y_pred_class)


# # ============================================================
# # 13. NEW SENTENCE INFERENCE
# # ============================================================

# x_predict = ["개똥이 잘생겼다"]


# # ============================================================
# # 1) TEXT → SEQUENCE
# # ============================================================

# x_predict = token.texts_to_sequences(x_predict)

# print("\n1. Sequence:")
# print(x_predict)
# # 1. Sequence:
# # [[24, 27]]

# # ============================================================
# # 2) PADDING
# # ============================================================

# x_predict = pad_sequences(
#     x_predict,
#     padding='pre',
#     maxlen=5,
#     truncating='post'
# )

# print("\n2. Padding:")
# print(x_predict) #  2. Padding:
#                  #  [[ 0  0  0 24 27]]

# print("Shape:", x_predict.shape)
# # Shape: (1, 5)


# # ============================================================
# # 3) LSTM INPUT SHAPE
# # ============================================================

# x_predict = x_predict.reshape(
#     x_predict.shape[0],
#     x_predict.shape[1],
#     1
# )

# print("\n3. LSTM Input Shape:")
# print(x_predict.shape)
# # (1, 5, 1)


# # ============================================================
# # 4) PREDICTION
# # ============================================================

# y_predict = model.predict(x_predict)

# print("\n4. Prediction Probability:")
# print(y_predict)
# #4. Prediction Probability:
# #[[0.36896586]]


# # ============================================================
# # 5) CLASSIFICATION
# # ============================================================

# y_predict_class = (y_predict > 0.5).astype(int)

# print("\n5. Prediction Class:")
# print(y_predict_class)
# # 5. Prediction Class:
# # [[0]]

# # ============================================================
# # 6) RESULT
# # ============================================================

# if y_predict_class[0][0] == 1:

#     print("\n[긍정]")
#     print(x_predict)

# else:

#     print("\n[부정]")
#     print(x_predict)
# #  #[부정]
# # [[[ 0]
# #   [ 0]
# #   [ 0]
# #   [24]
# #   [27]]]


# # # ============================================================
# # # 14. NEW SENTENCE INFERENCE
# # # ============================================================

# # new_sentence = [
# #     '개똥이 잘생겼다'
# # ]


# # # ------------------------------------------------------------
# # # 1) 새로운 문장을 Sequence로 변환
# # # ------------------------------------------------------------

# # new_x = token.texts_to_sequences(
# #     new_sentence
# # )


# # print("\n새로운 문장 Sequence:")
# # print(new_x)


# # # ------------------------------------------------------------
# # # 2) Padding
# # # ------------------------------------------------------------

# # new_x = pad_sequences(
# #     new_x,
# #     padding='pre',
# #     maxlen=5,
# #     truncating='post'
# # )


# # print("\n새로운 문장 Padding:")
# # print(new_x)


# # # ------------------------------------------------------------
# # # 3) LSTM 입력 형태로 변환
# # #
# # # (1, 5)
# # #    ↓
# # # (1, 5, 1)
# # # ------------------------------------------------------------

# # new_x = new_x.reshape(
# #     new_x.shape[0],
# #     new_x.shape[1],
# #     1
# # )


# # print("\nLSTM 입력 형태:")
# # print(new_x.shape)


# # # ------------------------------------------------------------
# # # 4) Prediction
# # # ------------------------------------------------------------

# # new_pred = model.predict(
# #     new_x
# # )


# # print("\n예측 확률:")
# # print(new_pred)


# # # ------------------------------------------------------------
# # # 5) Classification
# # # ------------------------------------------------------------

# # new_pred_class = (
# #     new_pred > 0.5
# # ).astype(int)


# # print("\n예측 결과:")
# # print(new_pred_class)


# # # ------------------------------------------------------------
# # # 6) 결과 출력
# # # ------------------------------------------------------------

# # if new_pred_class[0][0] == 1:

# #     print(
# #         "\n[긍정] 개똥이 잘생겼다."
# #     )

# # else:

# #     print(
# #         "\n[부정] 개똥이 잘생겼다."
# #     )

# # ##################################################################
# # # Training time : 1.8345069885253906

# # # ==============================
# # # Test Loss     : 1.082614779472351
# # # Test Accuracy : 0.0
# # # ==============================
# # # 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 44ms/step

# # # y_test:
# # # [0 1 1 0]

# # # y_pred:
# # # [[1]
# # #  [0]
# # #  [0]
# # #  [1]]

# # ########################################################################################################
# # #새로운 문장 Padding:

# # # [[ 0  0  0 24 27]]
# # # 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 41ms/step

# # # 예측 확률:
# # # [[0.08485556]]

# # # 예측 결과:
# # # [[0]]

# # # [부정] 개똥이 잘생겼다.

# # #####################################################################################

# # # 새로운 문장 Padding:
# # # [[ 0  0  0 24 27]]

# # # LSTM 입력 형태:
# # # (1, 5, 1)
# # # 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 90ms/step

# # # 예측 확률:
# # # [[0.36864364]]

# # # 예측 결과:
# # # [[0]]

# # # [부정] 개똥이 잘생겼다.
# ########################################################################################
# # 4. Prediction Probability:
# # [[0.36896586]]

# # 5. Prediction Class:
# # [[0]]

# # [부정]
# # [[[ 0]
# #   [ 0]
# #   [ 0]
# #   [24]
# #   [27]]]