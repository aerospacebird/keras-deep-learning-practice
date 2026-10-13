
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import random

print("TensorFlow version:", tf.__version__)

# 시드 고정
random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

# 1. 데이터
x = np.array([1,])
y = np.array([1,])

# 2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

# ============================================================
# ① 초기 가중치 출력 + 수동 순전파 계산
# ============================================================
print("\n" + "="*70)
print("① 초기 가중치 (훈련 전)")
print("="*70)

weights = model.get_weights()   # [W1, b1, W2, b2, W3, b3]

W1, b1 = weights[0], weights[1]
W2, b2 = weights[2], weights[3]
W3, b3 = weights[4], weights[5]

print("\n[Layer 1] W1 shape:", W1.shape)
print(W1)
print("b1:", b1)

print("\n[Layer 2] W2 shape:", W2.shape)
print(W2)
print("b2:", b2)

print("\n[Layer 3] W3 shape:", W3.shape)
print(W3)
print("b3:", b3)

# ----- 수동 순전파 (초기 가중치) -----
print("\n----- 초기 가중치로 수동 순전파 계산 -----")

# Layer 1
a1 = x.reshape(1, -1) @ W1 + b1          # (1,1) @ (1,3) + (3,) → (1,3)
print("Layer1 출력 (a1):", a1)

# Layer 2
a2 = a1 @ W2 + b2                        # (1,3) @ (3,2) + (2,) → (1,2)
print("Layer2 출력 (a2):", a2)

# Layer 3
y_hat_init = a2 @ W3 + b3                # (1,2) @ (2,1) + (1,) → (1,1)
print("최종 예측값 (초기):", y_hat_init)


# ============================================================
# ② 모델 컴파일 & 훈련
# ============================================================
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=500, verbose=0)   # verbose=0으로 조용히 훈련

print("\n" + "="*70)
print("훈련 완료 (500 epochs)")
print("="*70)


# ============================================================
# ③ 훈련 후 가중치 출력 + 수동 순전파 계산
# ============================================================
print("\n" + "="*70)
print("③ 훈련 후 가중치")
print("="*70)

weights_trained = model.get_weights()

W1_t, b1_t = weights_trained[0], weights_trained[1]
W2_t, b2_t = weights_trained[2], weights_trained[3]
W3_t, b3_t = weights_trained[4], weights_trained[5]

print("\n[Layer 1] W1:")
print(W1_t)
print("b1:", b1_t)

print("\n[Layer 2] W2:")
print(W2_t)
print("b2:", b2_t)

print("\n[Layer 3] W3:")
print(W3_t)
print("b3:", b3_t)

# ----- 수동 순전파 (훈련 후 가중치) -----
print("\n----- 훈련 후 가중치로 수동 순전파 계산 -----")

a1_t = x.reshape(1, -1) @ W1_t + b1_t
print("Layer1 출력 (a1):", a1_t)

a2_t = a1_t @ W2_t + b2_t
print("Layer2 출력 (a2):", a2_t)

y_hat_trained = a2_t @ W3_t + b3_t
print("최종 예측값 (훈련 후):", y_hat_trained)


# ============================================================
# ④ 모델의 predict 결과와 비교
# ============================================================
print("\n" + "="*70)
print("④ 모델.predict() 결과와 비교")
print("="*70)
print("model.predict(x):", model.predict(x, verbose=0))
print("수동 계산 결과  :", y_hat_trained)

######################################### ####################################################################

# ======================================================================
# ① 초기 가중치 (훈련 전)
# ======================================================================

# [Layer 1] W1 shape: (1, 3)
# [[ 0.13603318 -0.03480017  0.7743634 ]]
# b1: [0. 0. 0.]

# [Layer 2] W2 shape: (3, 2)
# [[-0.92561173  0.8256177 ]
#  [ 0.6200088   1.0182774 ]
#  [-0.5191052  -0.6304303 ]]
# b2: [0. 0.]

# [Layer 3] W3 shape: (2, 1)
# [[-0.02628279]
#  [-1.074922  ]]
# b3: [0.]

# ----- 초기 가중치로 수동 순전파 계산 -----
# Layer1 출력 (a1): [[ 0.13603318 -0.03480017  0.7743634 ]]
# Layer2 출력 (a2): [[-0.54946638 -0.41130697]]
# 최종 예측값 (초기): [[0.4565644]]
# 2026-10-13 12:53:17.971743: I tensorflow/stream_executor/cuda/cuda_blas.cc:1786] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.

# ======================================================================
# 훈련 완료 (500 epochs)
# ======================================================================

# ======================================================================
# ③ 훈련 후 가중치
# ======================================================================

# [Layer 1] W1:
# [[ 0.08450598 -0.08842857  0.8283193 ]]
# b1: [-0.05152716 -0.05362927  0.05395445]

# [Layer 2] W2:
# [[-0.9780322   0.78268254]
#  [ 0.6976038   1.0858626 ]
#  [-0.5848008  -0.68448573]]
# b2: [-0.06430288 -0.05273547]

# [Layer 3] W3:
# [[-0.08098927]
#  [-1.1340624 ]]
# b3: [0.05222854]

# ----- 훈련 후 가중치로 수동 순전파 계산 -----
# Layer1 출력 (a1): [[ 0.03297882 -0.14205784  0.88227376]]
# Layer2 출력 (a2): [[-0.71161171 -0.78508263]]
# 최종 예측값 (훈련 후): [[1.00019416]]

# ======================================================================
# ④ 모델.predict() 결과와 비교
# ======================================================================
# model.predict(x): [[0.99995005]]
# 수동 계산 결과  : [[1.00019416]]
# PS C:\study> 