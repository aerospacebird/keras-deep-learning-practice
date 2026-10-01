
import numpy as np
import time

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import Dense, Dropout

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# #1. 데이터
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
    '길동이 또 구라친다',
]

# y
y = np.array([
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 0,
    0, 1, 0, 1, 0
])


# ============================================================
# #2. Tokenizer
# ============================================================

token = Tokenizer()

token.fit_on_texts(docs)

print("=" * 70)
print("word_index")
print("=" * 70)

print(token.word_index)


# ============================================================
# #3. 문장 -> 정수 Sequence
# ============================================================

x = token.texts_to_sequences(docs)

print("\n" + "=" * 70)
print("Sequence")
print("=" * 70)

print(x)


# ============================================================
# #4. Padding
# ============================================================

x = pad_sequences(
    x,
    padding='pre',
    maxlen=5,
    truncating='post'
)

print("\n" + "=" * 70)
print("Padded x")
print("=" * 70)

print(x)

print("\nx.shape :", x.shape)


# ============================================================
# #5. 데이터 분리
# ============================================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.75,
    random_state=43,
    stratify=y
)

print("\n" + "=" * 70)
print("Data Split")
print("=" * 70)

print("x_train.shape :", x_train.shape)
print("x_test.shape  :", x_test.shape)
print("y_train.shape :", y_train.shape)
print("y_test.shape  :", y_test.shape)


# ============================================================
# #6. Embedding + DNN Model
# ============================================================

vocab_size = len(token.word_index) + 1

embedding_dim = 16

model = Sequential()


# ------------------------------------------------------------
# Embedding
# ------------------------------------------------------------

model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=5
    )
)


# ------------------------------------------------------------
# Embedding 결과를 하나의 벡터로 압축
# ------------------------------------------------------------

model.add(
    GlobalAveragePooling1D()
)


# ------------------------------------------------------------
# DNN
# ------------------------------------------------------------

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
    Dropout(0.2)
)


# ------------------------------------------------------------
# Binary Classification
# ------------------------------------------------------------

model.add(
    Dense(
        1,
        activation='sigmoid'
    )
)


# ============================================================
# #7. Model Compile
# ============================================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# ============================================================
# #8. Model Summary
# ============================================================

print("\n" + "=" * 70)
print("MODEL SUMMARY")
print("=" * 70)

model.summary()


# ============================================================
# #9. EarlyStopping
# ============================================================

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)


# ============================================================
# #10. Model Training
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

print("\nTraining time :", end - start)


# ============================================================
# #11. Test Evaluation
# ============================================================

loss, accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\n" + "=" * 70)
print("TEST RESULT")
print("=" * 70)

print("Test Loss     :", loss)
print("Test Accuracy :", accuracy)


# ============================================================
# #12. Test Prediction
# ============================================================

y_pred = model.predict(
    x_test,
    verbose=0
)

y_pred_class = (y_pred >= 0.5).astype(int)

print("\n" + "=" * 70)
print("TEST PREDICTION")
print("=" * 70)

print("y_test")
print(y_test)

print("\ny_pred probability")
print(y_pred.reshape(-1))

print("\ny_pred class")
print(y_pred_class.reshape(-1))


# ============================================================
# #13. 새로운 문장 여러 개
# ============================================================

new_docs = [
    '개똥이 잘생겼다',
    '개똥이 바보다',
    '말똥이 잘생겼다',
    '너무 재미있어요',
    '정말 재미없어요',
    '영화가 최고예요',
    '영화가 지루해요',
    '연기가 좋았어요',
    '연기가 어색해요',
    '추천하고 싶어요',
]


# ============================================================
# #14. 새로운 문장 -> Sequence
# ============================================================

new_x = token.texts_to_sequences(new_docs)

print("\n" + "=" * 70)
print("NEW SENTENCE SEQUENCE")
print("=" * 70)

for sentence, sequence in zip(new_docs, new_x):
    print(f"{sentence:20s} -> {sequence}")


# ============================================================
# #15. 새로운 문장 -> Padding
# ============================================================

new_x = pad_sequences(
    new_x,
    padding='pre',
    maxlen=5,
    truncating='post'
)

print("\n" + "=" * 70)
print("NEW SENTENCE PADDING")
print("=" * 70)

print(new_x)

print("\nnew_x.shape :", new_x.shape)


# ============================================================
# #16. 새로운 문장 Prediction
# ============================================================

new_pred = model.predict(
    new_x,
    verbose=0
)


# ============================================================
# #17. 0.5 기준으로 긍정 / 부정 분류
# ============================================================

new_pred_class = (new_pred >= 0.5).astype(int)


# ============================================================
# #18. 결과 출력
# ============================================================

print("\n" + "=" * 70)
print("NEW SENTENCE INFERENCE RESULT")
print("=" * 70)

for sentence, probability, prediction in zip(
    new_docs,
    new_pred.reshape(-1),
    new_pred_class.reshape(-1)
):

    if prediction == 1:
        result = "긍정"
    else:
        result = "부정"

    print(
        f"{sentence:20s} | "
        f"확률 = {probability:.4f} | "
        f"결과 = {result}"


    )

################################################################################################
# ======================================================================
# TEST RESULT
# ======================================================================
# Test Loss     : 0.692868709564209
# Test Accuracy : 0.5

# ======================================================================
# TEST PREDICTION
# ======================================================================
# y_test
# [0 1 1 0]

# y_pred probability
# [0.5011101  0.501775   0.50120455 0.501305  ]

# y_pred class
# [1 1 1 1]

# ======================================================================
# NEW SENTENCE SEQUENCE
# ======================================================================
# 개똥이 잘생겼다             -> [24, 27]
# 개똥이 바보다              -> [24]
# 말똥이 잘생겼다             -> [26, 27]
# 너무 재미있어요             -> [2]
# 정말 재미없어요             -> [21]
# 영화가 최고예요             -> [4]
# 영화가 지루해요             -> [18]
# 연기가 좋았어요             -> [19]
# 연기가 어색해요             -> [19, 20]
# 추천하고 싶어요             -> [7, 14]

# ======================================================================
# NEW SENTENCE PADDING
# ======================================================================
# [[ 0  0  0 24 27]
#  [ 0  0  0  0 24]
#  [ 0  0  0 26 27]
#  [ 0  0  0  0  2]
#  [ 0  0  0  0 21]
#  [ 0  0  0  0  4]
#  [ 0  0  0  0 18]
#  [ 0  0  0  0 19]
#  [ 0  0  0 19 20]
#  [ 0  0  0  7 14]]

# new_x.shape : (10, 5)

# ======================================================================
# NEW SENTENCE INFERENCE RESULT
# ======================================================================
# 개똥이 잘생겼다             | 확률 = 0.5009 | 결과 = 긍정
# 개똥이 바보다              | 확률 = 0.5012 | 결과 = 긍정
# 말똥이 잘생겼다             | 확률 = 0.5014 | 결과 = 긍정
# 너무 재미있어요             | 확률 = 0.5013 | 결과 = 긍정
# 정말 재미없어요             | 확률 = 0.5011 | 결과 = 긍정
# 영화가 최고예요             | 확률 = 0.5016 | 결과 = 긍정
# 영화가 지루해요             | 확률 = 0.5005 | 결과 = 긍정
# 연기가 좋았어요             | 확률 = 0.5020 | 결과 = 긍정
# 연기가 어색해요             | 확률 = 0.5018 | 결과 = 긍정
# 추천하고 싶어요             | 확률 = 0.5013 | 결과 = 긍정