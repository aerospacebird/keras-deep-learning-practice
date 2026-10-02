from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Embedding


(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=1000,
    #maxlen = 20,
    test_split=0.2,
)

print(x_train)
print(x_train.shape, y_train.shape) # (8982,) (8982,)
print(x_test.shape, y_test.shape)   # (2246,) (2246,)
print(y_train) # [ 3  4  3 ... 25  3 25]

print(np.unique(y_train)) #[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
                          # 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

print(type(x_train)) #<class 'numpy.ndarray'>
print(type(x_train[0])) # <class 'list'>

print(len(x_train[0]), len(x_train[1])) # 87 56

print("뉴스기사의 최대길이 :", max(len(i) for i in x_train)) # 2376
print("뉴스기사의 최소길이 :", min(len(i) for i in x_train)) # 13
print("뉴스기사의 평균길이 :", sum(map(len, x_train))/len(x_train)) # 145.5398574927633

# 뉴스기사의 최대길이 : 2376
# 뉴스기사의 최소길이 : 13
# 뉴스기사의 평균길이 : 145.5398574927633
#######################################################################################################
#  DATA 준비
num_words = 1000
num_classes = len(np.unique(y_train)) # y_train에 존재하는 class의 개수를 계산한다.

print()
print("num_words  :", num_words)
print("num_classes:", num_classes) #num_classes: 46

# 전처리 (패드 시퀀스)
maxlen = 100

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
print("After padding")
print("x_train.shape :", x_train.shape) # (8982, 100)
print("x_test.shape  :", x_test.shape)  # (2246, 100)

print()
print("x_train[0]")
print(x_train[0])

# x_train[0]
# [  0   0   0   0   0   0   0   0   0   0   0   0   0   1   2   2   8  43
#   10 447   5  25 207 270   5   2 111  16 369 186  90  67   7  89   5  19
#  102   6  19 124  15  90  67  84  22 482  26   7  48   4  49   8 864  39
#  209 154   6 151   6  83  11  15  22 155  11  15   7  48   9   2   2 504
#    6 258   6 272  11  15  22 134  44  11  15  16   8 197   2  90  67  52
#   29 209  30  32 132   6 109  15  17  12]

#exit()
# y 원핫 인코딩

y_train = to_categorical(
    y_train,
    num_classes=num_classes,
)

y_test = to_categorical(
    y_test,
    num_classes=num_classes,
)

print()
print("After One-Hot Encoding")
print("y_train.shape:", y_train.shape)  # y_train.shape: (8982, 46)
print("y_test.shape :", y_test.shape)   # y_test.shape : (2246, 46)
#exit()
print()
print("y_train[0]")
print(y_train[0])

#y_train[0]
# [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#exit()


#2.모델
from tensorflow.keras.layers import Dense,Embedding,SimpleRNN

model = Sequential()
# ################################################### Embedding 1 ################################################# 
# model.add(Embedding(input_dim=1000, output_dim=46, input_length=100))
#                    # 단어사전의 갯수,      차원,
# model.add(SimpleRNN(100))
# model.add(Dense(46, activation='softmax'))

################################################### Embedding 2 ################################################# 
model.add(Embedding(input_dim=1000, output_dim=46, input_length=100))
                   # 단어사전의 갯수,      차원,
model.add(GRU(1000))
model.add(Dense(46, activation='softmax'))



#3. 컴파일 훈련
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.fit(
    x_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

# ============================================================
# 4. MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# 5. TRAINING
# ============================================================

# history = model.fit(
#     x_train,
#     y_train,
#     epochs=100,
#     batch_size=32,
#     validation_split=0.2,
#     verbose=1
# )


# ============================================================
# 6. EVALUATION
# ============================================================

loss, acc = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print()
print("Test Loss     :", loss)
print("Test Accuracy :", acc)


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