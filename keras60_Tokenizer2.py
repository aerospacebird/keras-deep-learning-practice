import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical

text1 = " 나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다."
text2 = "개똥이는 기관사를 좋아한다. 말똥이는 잘생겼다. 길동이는 마구 마구 더 잘생겼다. "

token = Tokenizer() # 객체(인스턴스) = class(), 인스턴스 생성.
token.fit_on_texts([text1, text2])

print(token.word_index)
#{'마구': 1, '진짜': 2, '매우': 3, '잘생겼다': 4, '나는': 5, '지금': 6, '맛있는': 7, '김밥을': 8, '엄청': 9, '먹었다': 10, '개똥이는': 11, '기관사를': 12, '좋아한다': 13, '말똥이는': 14, '길동이는': 15, '더': 16}

x = token.texts_to_sequences([text1, text2])

print(x)
##[[5, 6, 2, 2, 3, 3, 7, 8, 9, 1, 1, 1, 1, 10], [11, 12, 13, 14, 4, 15, 1, 1, 16, 4]]

#x= np.concatenate()

# ============================================================
# 5. 두 문장의 sequence를 하나로 연결
# ============================================================

x = np.concatenate(x)

print("\n===== np.concatenate(x) =====")

print(x)

print("\nshape :", x.shape)
# ===== np.concatenate(x) =====
# [ 5  6  2  2  3  3  7  8  9  1  1  1  1 10 11 12 13 14  4 15  1  1 16  4]

# shape : (24,)

# ============================================================
# 6. 정수 sequence 확인
# ============================================================

print("\n===== 전체 토큰 개수 =====")

print(len(x))

# ===== 전체 토큰 개수 =====
# 24

# ============================================================
# 7. One-Hot Encoding 방법 ①
#    TensorFlow / Keras
#    to_categorical()
# ============================================================

print("\n\n========================================")
print("① Keras to_categorical")
print("========================================")

# Tokenizer의 index는 1부터 시작
# 0은 사용하지 않으므로 num_classes는 vocabulary size + 1
num_classes = len(token.word_index) + 1

x_ohe_keras = to_categorical(
    x,
    num_classes=num_classes
)

print(x_ohe_keras)

print("\nshape :", x_ohe_keras.shape)

# ========================================
# ① Keras to_categorical
# ========================================
# [[0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

# shape : (24, 17)


# ============================================================
# 8. One-Hot Encoding 방법 ②
#    Scikit-Learn
#    OneHotEncoder()
# ============================================================

print("\n\n========================================")
print("② Scikit-Learn OneHotEncoder")
print("========================================")

# OneHotEncoder는 2차원 입력을 요구한다.
# 따라서 (N,) → (N,1)로 reshape
x_2d = x.reshape(-1, 1)

print("reshape 후 shape :", x_2d.shape)
# reshape 후 shape : (24, 1)

# 최신 sklearn에서는 sparse_output=False 사용
ohe = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

x_ohe_sklearn = ohe.fit_transform(x_2d)

print("\nOne-Hot Encoding 결과")

print(x_ohe_sklearn)

print("\nshape :", x_ohe_sklearn.shape)
#shape : (24, 16)

print("\nEncoder categories_")
#Encoder categories_
[array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16])]

print(ohe.categories_)

# Encoder categories_
# [array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16])]

# ========================================
# ② Scikit-Learn OneHotEncoder
# ========================================
# reshape 후 shape : (24, 1)

# One-Hot Encoding 결과
# [[0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

# shape : (24, 16)


# ============================================================
# 9. One-Hot Encoding 방법 ③
#    Pandas get_dummies()
# ============================================================

print("\n\n========================================")
print("③ Pandas get_dummies")
print("========================================")

# Series 형태로 변환
x_series = pd.Series(x)

x_ohe_pandas = pd.get_dummies(
    x_series
)

print(x_ohe_pandas)

print("\nshape :", x_ohe_pandas.shape)

# ========================================
# ③ Pandas get_dummies
# ========================================
#        1      2      3      4      5      6      7      8      9      10     11     12     13     14     15     16
# 0   False  False  False  False   True  False  False  False  False  False  False  False  False  False  False  False
# 1   False  False  False  False  False   True  False  False  False  False  False  False  False  False  False  False
# 2   False   True  False  False  False  False  False  False  False  False  False  False  False  False  False  False
# 3   False   True  False  False  False  False  False  False  False  False  False  False  False  False  False  False
# 4   False  False   True  False  False  False  False  False  False  False  False  False  False  False  False  False
# 5   False  False   True  False  False  False  False  False  False  False  False  False  False  False  False  False
# 6   False  False  False  False  False  False   True  False  False  False  False  False  False  False  False  False
# 7   False  False  False  False  False  False  False   True  False  False  False  False  False  False  False  False
# 8   False  False  False  False  False  False  False  False   True  False  False  False  False  False  False  False
# 9    True  False  False  False  False  False  False  False  False  False  False  False  False  False  False  False
# 10   True  False  False  False  False  False  False  False  False  False  False  False  False  False  False  False
# 11   True  False  False  False  False  False  False  False  False  False  False  False  False  False  False  False
# 12   True  False  False  False  False  False  False  False  False  False  False  False  False  False  False  False
# 13  False  False  False  False  False  False  False  False  False   True  False  False  False  False  False  False
# 14  False  False  False  False  False  False  False  False  False  False   True  False  False  False  False  False
# 15  False  False  False  False  False  False  False  False  False  False  False   True  False  False  False  False
# 16  False  False  False  False  False  False  False  False  False  False  False  False   True  False  False  False
# 17  False  False  False  False  False  False  False  False  False  False  False  False  False   True  False  False
# 18  False  False  False   True  False  False  False  False  False  False  False  False  False  False  False  False
# 19  False  False  False  False  False  False  False  False  False  False  False  False  False  False   True  False
# 20   True  False  False  False  False  False  False  False  False  False  False  False  False  False  False  False
# 21   True  False  False  False  False  False  False  False  False  False  False  False  False  False  False  False
# 22  False  False  False  False  False  False  False  False  False  False  False  False  False  False  False   True
# 23  False  False  False   True  False  False  False  False  False  False  False  False  False  False  False  False

# shape : (24, 16)



# ============================================================
# 10. 결과 비교
# ============================================================

print("\n\n========================================")
print("결과 비교")
print("========================================")

print("원본 x                :", x.shape)

print("Keras                 :", x_ohe_keras.shape)

print("Scikit-Learn          :", x_ohe_sklearn.shape)

print("Pandas                :", x_ohe_pandas.shape)

# ========================================
# 결과 비교
# ========================================
# 원본 x                : (24,)
# Keras                 : (24, 17)
# Scikit-Learn          : (24, 16)
# Pandas                : (24, 16)

#############################################################################################
# ===== np.concatenate(x) =====
# [ 5  6  2  2  3  3  7  8  9  1  1  1  1 10 11 12 13 14  4 15  1  1 16  4]

# shape : (24,)

# ===== 전체 토큰 개수 =====
# 24


# ========================================
# ① Keras to_categorical
# ========================================
# [[0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

# shape : (24, 17)


# ========================================
# ② Scikit-Learn OneHotEncoder
# ========================================
# reshape 후 shape : (24, 1)

# One-Hot Encoding 결과
# [[0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

# shape : (24, 16)

# Encoder categories_
# [array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16])]


# ========================================
# ③ Pandas get_dummies
# ========================================
#        1      2      3      4      5      6      7      8      9      10     11     12     13     14     15     16
# 0   False  False  False  False   True  False  False  False  False  False  False  False  False  False  False  False
# 1   False  False  False  False  False   True  False  False  False  False  False  False  False  False  False  False
# 2   False   True  False  False  False  False  False  False  False  False  False  False  False  False  False  False
# 3   False   True  False  False  False  False  False  False  False  False  False  False  False  False  False  False
# 4   False  False   True  False  False  False  False  False  False  False  False  False  False  False  False  False
# 5   False  False   True  False  False  False  False  False  False  False  False  False  False  False  False  False
# 6   False  False  False  False  False  False   True  False  False  False  False  False  False  False  False  False
# 7   False  False  False  False  False  False  False   True  False  False  False  False  False  False  False  False
# 8   False  False  False  False  False  False  False  False   True  False  False  False  False  False  False  False
# 9    True  False  False  False  False  False  False  False  False  False  False  False  False  False  False  False
# 10   True  False  False  False  False  False  False  False  False  False  False  False  False  False  False  False
# 11   True  False  False  False  False  False  False  False  False  False  False  False  False  False  False  False
# 12   True  False  False  False  False  False  False  False  False  False  False  False  False  False  False  False
# 13  False  False  False  False  False  False  False  False  False   True  False  False  False  False  False  False
# 14  False  False  False  False  False  False  False  False  False  False   True  False  False  False  False  False
# 15  False  False  False  False  False  False  False  False  False  False  False   True  False  False  False  False
# 16  False  False  False  False  False  False  False  False  False  False  False  False   True  False  False  False
# 17  False  False  False  False  False  False  False  False  False  False  False  False  False   True  False  False
# 18  False  False  False   True  False  False  False  False  False  False  False  False  False  False  False  False
# 19  False  False  False  False  False  False  False  False  False  False  False  False  False  False   True  False
# 20   True  False  False  False  False  False  False  False  False  False  False  False  False  False  False  False
# 21   True  False  False  False  False  False  False  False  False  False  False  False  False  False  False  False
# 22  False  False  False  False  False  False  False  False  False  False  False  False  False  False  False   True
# 23  False  False  False   True  False  False  False  False  False  False  False  False  False  False  False  False

# shape : (24, 16)


# ========================================
# 결과 비교
# ========================================
# 원본 x                : (24,)
# Keras                 : (24, 17)
# Scikit-Learn          : (24, 16)
# Pandas                : (24, 16)