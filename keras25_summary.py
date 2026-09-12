from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

"""
#######################################################################################################################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 모델 생성
model = Sequential()

model.add(Dense(64, input_dim=4, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))


# =========================================================
# Parameter 개수 직접 계산
# 공식:
# Parameter = (입력 뉴런 수 + 1) × 출력 뉴런 수
#             ↑
#             Bias
# =========================================================

# Layer 1 : Input 4 → Dense 64
param1 = (4 + 1) * 64

# Layer 2 : Dense 64 → Dense 32
param2 = (64 + 1) * 32

# Layer 3 : Dense 32 → Dense 10
param3 = (32 + 1) * 10

# 전체 Parameter
total_params = param1 + param2 + param3

print("Layer 1 Parameter :", param1) #Layer 1 Parameter : 320
print("Layer 2 Parameter :", param2) #Layer 2 Parameter : 2080
print("Layer 3 Parameter :", param3) #Layer 3 Parameter : 330
print("Total Parameter   :", total_params) #Total Parameter   : 2730


# Keras가 계산한 Parameter와 비교
print("\n===== Keras model.summary() =====")

#Layer 1 Parameter : 320
#Layer 2 Parameter : 2080
#Layer 3 Parameter : 330
#Total Parameter   : 2730

#exit()
# model.summary()



#######################################################################################################################################
"""
#2. 모델
model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))



# =========================================================
# Parameter 개수 직접 계산
# 공식:
# Parameter = (입력 뉴런 수 + 1) × 출력 뉴런 수
#             ↑
#             Bias
# =========================================================

# Layer 1 : Input 4 → Dense 64
param1 = (4 + 1) * 64

# Layer 2 : Dense 64 → Dense 32
param2 = (64 + 1) * 32

# Layer 3 : Dense 32 → Dense 10
param3 = (32 + 1) * 10

# 전체 Parameter
total_params = param1 + param2 + param3

print("Layer 1 Parameter :", param1) #Layer 1 Parameter : 320
print("Layer 2 Parameter :", param2) #Layer 2 Parameter : 2080
print("Layer 3 Parameter :", param3) #Layer 3 Parameter : 330
print("Total Parameter   :", total_params) #Total Parameter   : 2730
# ROC/AUC는 특히 이진 분류(Binary Classification) 모델의 성능을 평가할 때 사용하는 대표적인 지표
#ROC = Receiver Operating Characteristic Curve

#모델의 분류 임계값(threshold)을 바꿔가면서 다음 두 값을 계산하여 그린 그래프입니다.

#TPR (True Positive Rate) = 실제 양성을 얼마나 잘 찾아내는가
#FPR (False Positive Rate) = 실제 음성을 양성이라고 잘못 판단하는 비율

#모델의 분류 임계값(threshold)을 바꿔가면서 다음 두 값을 계산하여 그린 그래프입니다.

#AUC = Area Under the Curve

# 즉,ROC 곡선 아래의 면적 입니다.  AUC는 0~1 사이의 값을 가지며,
# 핵심적으로 AUC가 높을수록 양성과 음성을 잘 구분하는 모델입니다.

model.summary()

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 실제 정답
y_true = y_test

# 모델의 확률값
y_pred_proba = model.predict(x_test).ravel()

# ROC 계산
fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

# AUC 계산
roc_auc = auc(fpr, tpr)

print("AUC :", roc_auc)

# ROC Curve
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")

# Random Classifier
plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()