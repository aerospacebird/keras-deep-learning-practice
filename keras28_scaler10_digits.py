from sklearn.datasets import load_wine
# acc = 0.95 이상을 만들어 보세요?
from sklearn.datasets import load_digits

# acc: 1.0
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
 
from tensorflow.keras.utils import to_categorical



#1. DATA준비

datasets = load_digits()    #머신러닝의 다중분류(Multi-class Classification)

print(datasets)
print(datasets.data)
print(datasets.DESCR)
print(datasets.feature_names)

#exit()

# Number of Instances: 1797
#:Number of Attributes: 64
x = datasets.data
y = datasets['target']

print(x.shape, y.shape)  #(1797, 64) (1797,)
print(y)
print(np.unique(y, return_counts=True))  # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]))

#exit()
"""
[0,0,1,1,2]   #(5, 0)
---------------------->>>>>>>>>>>>>>>>>>>>
[[1,0,0],
[1,0,0],
[0,1,0],
[0,1,0],
[0,0,1]]       #(5, 3)
"""

####################################################### 원핫 1. to_catogorical ####################################################
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)  #(1797, 10)    one hot encoding한후에 split 시켜라!!!  아니면 2번 연속 반복해야 한다.
#exit()
"""
####################################################### 원핫 2. pandas   #############################################################
# pandas get_dummies() 함수 이용
df_dummies = pd.get_dummies(data=df, columns=cols_o)
df_dummies

####################################################### 원핫 3. sklearn ##############################################################
# scikit learn OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
arr_ohe = ohe.fit_transform(df[['X1', 'X2', 'X3']])
arr_ohe
######################################################################################################################################
"""
"""
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical

# ============================================================
# 1. Iris DATA
# ============================================================

datasets = load_iris()

x = datasets.data
y = datasets.target

print("원래 y")
print(y)
print("원래 y shape :", y.shape)


# ============================================================
# 2. 방법 1 : TensorFlow to_categorical()
# ============================================================

y_keras = to_categorical(y)

print("\n[방법 1] to_categorical()")
print(y_keras)
print("shape :", y_keras.shape)


# ============================================================
# 3. 방법 2 : Pandas get_dummies()
# ============================================================

df = pd.DataFrame({
    'y': y
})

df_dummies = pd.get_dummies(
    data=df,
    columns=['y']
)

y_pandas = df_dummies.values.astype(float)

print("\n[방법 2] pandas get_dummies()")
print(y_pandas)
print("shape :", y_pandas.shape)


# ============================================================
# 4. 방법 3 : sklearn OneHotEncoder()
# ============================================================

y_reshape = y.reshape(-1, 1)

ohe = OneHotEncoder(
    sparse_output=False
)

y_sklearn = ohe.fit_transform(y_reshape)

print("\n[방법 3] sklearn OneHotEncoder()")
print(y_sklearn)
print("shape :", y_sklearn.shape)


# ============================================================
# 5. 세 가지 결과 비교
# ============================================================

print("\n==============================")
print("세 가지 결과 비교")
print("==============================")

print("to_categorical :", y_keras.shape)
print("get_dummies    :", y_pandas.shape)
print("OneHotEncoder  :", y_sklearn.shape)

print("\n방법 1 == 방법 2 :", np.array_equal(y_keras, y_pandas))
print("방법 1 == 방법 3 :", np.array_equal(y_keras, y_sklearn))
print("방법 2 == 방법 3 :", np.array_equal(y_pandas, y_sklearn))

#shape : (150, 3)
"""
"""
==============================
세 가지 결과 비교
==============================
to_categorical : (150, 3)
get_dummies    : (150, 3)
OneHotEncoder  : (150, 3)

방법 1 == 방법 2 : True
방법 1 == 방법 3 : True
방법 2 == 방법 3 : True

"""
#exit()

x_train, x_test, y_train, y_test = train_test_split(
                                x, y,
                                train_size=0.8,
                                random_state=42,
                                shuffle=True,
                                stratify=y,                       
)

print(x_train.shape, x_test.shape) # (142, 13) (36, 13)
print(y_train.shape, y_test.shape) # (142, 3) (36, 3)

#exit()
####################################################################################################################
from sklearn.preprocessing import MinMaxScaler , StandardScaler, MaxAbsScaler, RobustScaler
#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = MaxAbsScaler()
scaler = RobustScaler() # **RobustScaler()**는 scikit-learn의 전처리 클래스(sklearn.preprocessing.RobustScaler)로, 
                        # 이상치(outlier)에 강건한 방식으로 특성을 스케일링합니다.
                        # **중앙값(median)**과 **사분위수 범위(IQR)**를 사용합니다.
                        #  IQR = 75번째 백분위수 − 25번째 백분위수 (사분위 범위)
#scaler.fit(x_train)  #준비
#x_train = scaler.transform(x_train)  #변환

x_train = scaler.fit_transform(x_train)  #변환
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

#0.0 1.0
#0.0 1.0666666666666667
#exit()

#######################################################################################################################
"""
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam

# 핵심 목적은 x_train과 x_test의 각 특성(feature)을 비슷한 스케일로 맞춰서 머신러닝/딥러닝 모델이 안정적으로 학습하도록 하는 것

# Scaling
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)   #Train과 Test가 서로 다른 기준으로 Scaling됩니다.훈련 데이터의 평균과 표준편차를 계산
x_test = scaler.transform(x_test)    # 계산한 평균과 표준편차로 데이터를 변환

StandardScaler is a preprocessing tool in the Python library scikit-learn that standardizes features by removing the mean and scaling to unit variance.

The transformation is:

$$ z = \frac{x - \mu}{\sigma} $$

where:

\(x\) = original value
\(\mu\) = mean of the feature
\(\sigma\) = standard deviation of the feature
\(z\) = standardized value

After scaling:

Mean ≈ 0
Standard deviation ≈ 1
Why use StandardScaler?

Many machine learning algorithms perform better when features are on a similar scale, including:

Logistic Regression
Linear Regression (with regularization)
Support Vector Machines (SVM)
K-Nearest Neighbors (KNN)
Neural Networks
PCA

It is generally not necessary for tree-based models like:

Decision Trees
Random Forests
Gradient Boosting (XGBoost, LightGBM, CatBoost)

"""
#2. 모델구성
model = Sequential()
model.add(Dense(128, input_dim=64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))  # activation='softmax'는 출력층의 값을 확률 분포 형태로 변환하는 활성화 함수
                                           # 각 클래스에 속할 확률을 출력하고, 모든 클래스의 확률 합은 1이 됩니다.

#3. 컴파일  &  훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])  # sparse_categorical_crossentropy 정수 형태의 라벨이라면
es= EarlyStopping(                                                                        # 일반적인 loss='categorical_crossentropy'
    monitor= 'val_loss',
    mode= 'auto',
    patience=500,
    restore_best_weights=True,
)

start_time= time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=64,
          verbose=1,
          validation_split=0.2,
          callbacks=[es],
)
end_time = time.time()

#4. 평가 및 추론
result = model.evaluate(x_test, y_test)
print('loss :', result[0])
print('acc:', round(result[1], 2))

y_predict = model.predict(x_test)
print(y_predict)

y_predict = np.argmax(y_predict, axis=1) #axis=1은 **2차원 배열에서 "각 행(row)을 기준으로 계산하라"**는 의미입니다.
print(y_predict)  #[0 2 0 2 1 1 0 2 0 2 2 2 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 2]
#  argmax 핵심 이유는 One-Hot Encoding된 y_test를 다시 원래의 정답 클래스 번호로 바꾸기 위해서?
y_test = np.argmax(y_test, axis=1) #axis=1은 **2차원 배열에서 "각 행(row)을 기준으로 계산하라"**는 의미입니다.
print(y_test)  # [0 2 0 1 1 1 0 2 0 2 2 2 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 1]



accuracy_score = accuracy_score(y_test, y_predict)
print('acc_score:', accuracy_score)
print('걸린시간:', round(end_time - start_time, 2), '초')



# loss : 0.1494787633419037
# acc: 0.96

#acc_score: 0.9638888888888889
#걸린시간: 4.33 초

#loss : 0.13612346351146698
#acc: 0.97

#acc_score: 0.9722222222222222
#걸린시간: 183.75 초

#loss : 0.1363675445318222
#acc: 0.98

#acc_score: 0.975
#걸린시간: 37.55 초

#loss : 0.11519830673933029
#acc: 0.97

#acc_score: 0.9694444444444444
#걸린시간: 37.8 초

#loss : 0.3127055764198303 #StandardScaler
#acc: 0.97                 #StandardScaler

#acc_score: 0.9722222222222222  #StandardScaler
#걸린시간: 76.34 초              #StandardScaler

#loss : 0.39555639028549194    #MaxAbsScaler

#loss : 0.1498822420835495  # RobustScaler
#acc: 0.96

#acc_score: 0.9611111111111111  # RobustScaler
#걸린시간: 38.64 초

"""
              출력 활성화        손실 함수
────────────────────────────────────────────
회귀          Linear             MSE / MAE          
이진분류       Sigmoid           Binary CE                       np.round()
다중분류       Softmax           Categorical CE (OHE)            np.argmax()
다중분류       Softmax           Sparse Categorical CE(NON_OHE)
                              ↑
                         One-Hot 여부

"""

#실무적인 선택 기준으로는:

#데이터에 이상치가 많거나 정규분포에 가까운 경우 → StandardScaler를 우선 고려

#입력값을 특정 범위(특히 0~1)로 맞추는 것이 중요한 경우 → MinMaxScaler를 고려