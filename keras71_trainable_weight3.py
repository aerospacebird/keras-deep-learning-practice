

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import random

print(tf.__version__)

random.seed(333) #초기값을 고정시켜준다.
np.random.seed(333) #초기값을 고정시켜준다.
tf.random.set_seed(333)  #초기값을 고정시켜준다.

#1. DATA 준비
x = np.array([1,])
y = np.array([1,])


#2. MODEL
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  dense (Dense)               (None, 3)                 6          #  (INPUT + 1)* OUTPUT= TOTAL PARAMETERS
                                                                 
#  dense_1 (Dense)             (None, 2)                 8         
                                                                 
#  dense_2 (Dense)             (None, 1)                 3         
                                                                 
# =================================================================
# Total params: 17
# Trainable params: 17
# Non-trainable params: 0
# _________________________________________________________________
print("========================================= model.weights===========================================")
print(model.weights)
print("========================================= model.trainable weights=================================")
print(model.trainable_weights)

# [<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.13603318, -0.03480017,  0.7743634 ]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
# array([[-0.92561173,  0.8256177 ],
#        [ 0.6200088 ,  1.0182774 ],
#        [-0.5191052 , -0.6304303 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
# array([[-0.02628279],
#        [-1.074922  ]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
###############################################################################################################################################
# ========================================= model.trainable weights=================================
# [<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.13603318, -0.03480017,  0.7743634 ]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
# array([[-0.92561173,  0.8256177 ],
#        [ 0.6200088 ,  1.0182774 ],
#        [-0.5191052 , -0.6304303 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
# array([[-0.02628279],
#        [-1.074922  ]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
# print("=======================================================================================================")
# print(len(model.weights)) #6
# print(len(model.trainable_weights)) #6

############################################ 동결 #################################################################
#model.trainable = False
model.trainable = True

# print("=======================================================================================================")
# print(model.weights)
# print("=======================================================================================================")
# print(len(model.weights)) #6
# print(len(model.trainable_weights)) #0

# print("=======================================================================================================")
# print(model.trainable_weights)
# print("=======================================================================================================")
# print(model.weights)

#model.summary()

# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  dense (Dense)               (None, 3)                 6         
                                                                 
#  dense_1 (Dense)             (None, 2)                 8         
                                                                 
#  dense_2 (Dense)             (None, 1)                 3         
                                                                 
# =================================================================
# Total params: 17
# Trainable params: 0   # NPU  모델 가중치 고정.
# Non-trainable params: 17
# _________________________________________________________________
#exit()
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=500, verbose=1)

#4. 평가, 예측
y_predict= model.predict(x)
print(y_predict)

##########################################################################################################
# [[1.2028586]
#  [2.1229634]
#  [3.0427358]
#  [3.961594 ]
#  [4.8808894]]
################################################## 동결 ##################################################
# [[0.4565795]
#  [0.913159 ]
#  [1.3704137]
#  [1.826318 ]
#  [2.283362 ]]
##########################################################################################################
# print("=======================================================================================================")
# print(model.weights)

# #################################################################################################################
# =======================================================================================================
# [<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.08450598, -0.08842857,  0.8283193 ]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([-0.05152716, -0.05362927,  0.05395445], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
# array([[-0.9780322 ,  0.78268254],
#        [ 0.6976038 ,  1.0858626 ],
#        [-0.5848008 , -0.68448573]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([-0.06430288, -0.05273547], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
# array([[-0.08098927],
#        [-1.1340624 ]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.05222854], dtype=float32)>]
#################################################################################################################
# """
# 6. 훈련 후 최종 가중치
# 코드 맨 아래에 출력된 값이 바로 500 epoch 후의 가중치입니다.
# PythonCopyCopieddense/kernel:0   → [[ 0.08450598, -0.08842857,  0.8283193 ]]
# dense/bias:0     → [-0.05152716, -0.05362927,  0.05395445]
# dense_1/kernel:0 → [[-0.9780322 ,  0.78268254],
#                      [ 0.6976038 ,  1.0858626 ],
#                      [-0.5848008 , -0.68448573]]
# dense_1/bias:0   → [-0.06430288, -0.05273547]
# dense_2/kernel:0 → [[-0.08098927],
#                      [-1.1340624 ]]
# dense_2/bias:0   → [0.05222854]
# 초기값과 비교하면 모든 값이 조금씩 이동한 것을 볼 수 있습니다.
# 특히 bias들이 0에서 벗어나기 시작한 것이 특징입니다.
# 요약 (가중치 계산 전체 과정)

# 초기화 : Glorot Uniform + Bias=0 (시드 고정으로 재현 가능)
# 순전파 : $\hat{y} = ((x W_1 + b_1) W_2 + b_2) W_3 + b_3$
# 손실 계산 : MSE
# 역전파 : 자동 미분으로 17개 파라미터에 대한 기울기 계산
# Adam 업데이트 : 500번 반복 → 최종 가중치 획득

# model.trainable = False로 바꾸면 4~5번 과정이 아예 일어나지 않고, 가중치가 초기값 그대로 유지됩니다 (코드에 주석 처리된 부분).
# 더 궁금한 부분(예: 특정 레이어의 기울기 수식, Adam 하이퍼파라미터 영향 등)이 있으면 말씀해주세요!모든 레이어를 계산하라Worked for 17s모든 레이어의 순전파(Forward Pass) 계산을 초기 가중치와 훈련 후 가중치 두 경우로 나눠서 정확히 계산해 드리겠습니다.
# 입력: $x = 1$

# 1. 초기 가중치로 계산 (훈련 전)
# Layer 1 (Dense 3)
# $$\begin{align*}
# W_1 &= \begin{bmatrix} 0.13603318 & -0.03480017 & 0.7743634 \end{bmatrix} \\
# b_1 &= \begin{bmatrix} 0 & 0 & 0 \end{bmatrix}
# \end{align*}$$
# $$a_1 = x \cdot W_1 + b_1 = 1 \cdot W_1 + b_1 = \begin{bmatrix} 0.13603318 & -0.03480017 & 0.7743634 \end{bmatrix}$$
# Layer 2 (Dense 2)
# $$W_2 = \begin{bmatrix}
# -0.92561173 & 0.8256177 \\
# 0.6200088 & 1.0182774 \\
# -0.5191052 & -0.6304303
# \end{bmatrix}, \quad
# b_2 = \begin{bmatrix} 0 & 0 \end{bmatrix}$$
# $$\begin{align*}
# a_2 &= a_1 \cdot W_2 + b_2 \\
# &= \begin{bmatrix} 0.13603318 & -0.03480017 & 0.7743634 \end{bmatrix}
# \begin{bmatrix}
# -0.92561173 & 0.8256177 \\
# 0.6200088 & 1.0182774 \\
# -0.5191052 & -0.6304303
# \end{bmatrix} \\
# &= \begin{bmatrix} -0.54946639 & -0.41130698 \end{bmatrix}
# \end{align*}$$
# Layer 3 (Dense 1) – 최종 출력
# $$W_3 = \begin{bmatrix} -0.02628279 \\ -1.074922 \end{bmatrix}, \quad
# b_3 = \begin{bmatrix} 0 \end{bmatrix}$$
# $$\begin{align*}
# \hat{y} &= a_2 \cdot W_3 + b_3 \\
# &= \begin{bmatrix} -0.54946639 & -0.41130698 \end{bmatrix}
# \begin{bmatrix} -0.02628279 \\ -1.074922 \end{bmatrix} \\
# &= \mathbf{0.45656443}
# \end{align*}$$

# 2. 훈련 후 가중치로 계산 (500 epoch 후)
# Layer 1 (Dense 3)
# $$\begin{align*}
# W_1 &= \begin{bmatrix} 0.08450598 & -0.08842857 & 0.8283193 \end{bmatrix} \\
# b_1 &= \begin{bmatrix} -0.05152716 & -0.05362927 & 0.05395445 \end{bmatrix}
# \end{align*}$$
# $$a_1 = 1 \cdot W_1 + b_1 = \begin{bmatrix} 0.03297882 & -0.14205784 & 0.88227375 \end{bmatrix}$$
# Layer 2 (Dense 2)
# $$W_2 = \begin{bmatrix}
# -0.9780322 & 0.78268254 \\
# 0.6976038 & 1.0858626 \\
# -0.5848008 & -0.68448573
# \end{bmatrix}, \quad
# b_2 = \begin{bmatrix} -0.06430288 & -0.05273547 \end{bmatrix}$$
# $$a_2 = a_1 \cdot W_2 + b_2 = \begin{bmatrix} -0.71161171 & -0.78508261 \end{bmatrix}$$
# Layer 3 (Dense 1) – 최종 출력
# $$W_3 = \begin{bmatrix} -0.08098927 \\ -1.1340624 \end{bmatrix}, \quad
# b_3 = \begin{bmatrix} 0.05222854 \end{bmatrix}$$
# $$\hat{y} = a_2 \cdot W_3 + b_3 = \mathbf{1.00019412}$$

# 요약 비교
# 구분Layer 1 출력Layer 2 출력최종 예측 $\hat{y}$초기 가중치[0.1360, -0.0348, 0.7744][-0.5495, -0.4113]0.4566훈련 후[0.0330, -0.1421, 0.8823][-0.7116, -0.7851]1.0002
# 목표값 $y = 1$에 거의 정확히 맞춰진 것을 확인할 수 있습니다.
# (소수점 오차는 부동소수점 연산과 Adam의 업데이트 과정에서 생긴 정상적인 수치입니다.)


# """