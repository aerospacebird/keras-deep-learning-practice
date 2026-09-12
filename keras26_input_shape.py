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

datasets = load_iris()    #머신러닝의 다중분류(Multi-class Classification)
print(datasets)
print(datasets.data)
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets['target']

print(x.shape, y.shape)  #(150, 4) (150,)
print(y)
print(np.unique(y, return_counts=True))  # (array([0, 1, 2]), array([50, 50, 50]))
"""
[0,0,1,1,2]   #(5, 0)
---------------------->>>>>>>>>>>>>>>>>>>>
[[1,0,0],
[1,0,0],
[0,1,0],
[0,1,0],
[0,0,1]]       #(5, 3)
"""
"""
####################################################### 원핫 1. to_catogorical ####################################################
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)  #(150, 3)    one hot encoding한후에 split 시켜라!!!  아니면 2번 연속 반복해야 한다.
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
"""
y_keras = to_categorical(y)

print("\n[방법 1] to_categorical()")
print(y_keras)
print("shape :", y_keras.shape)

"""
# ============================================================
# 3. 방법 2 : Pandas get_dummies()
# ============================================================
"""
y = pd.get_dummies(y, dtype=int)
print(y)
print(y.shape)










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

"""

# ============================================================
# 4. 방법 3 : sklearn OneHotEncoder()
# ============================================================
from sklearn.preprocessing import OneHotEncoder
y = y.reshape(150, 1)
y = y.reshape(-1, 1)

print(y, y.shape)
#exit()
#y_reshape = y.reshape(-1, 1)

#ohe = OneHotEncoder() # sparse 형태로 나온다.
ohe = OneHotEncoder(sparse_output=False)

y = ohe.fit_transform(y)
#y= ohe.fit_transform(y)
print(y)
#exit()


#    sparse_output=False
#)

#y_sklearn = ohe.fit_transform(y_reshape)

print("\n[방법 3] sklearn OneHotEncoder()")
print(y)
print("shape :", y.shape)


# ============================================================
# 5. 세 가지 결과 비교
# ============================================================
"""
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
#==============================
#세 가지 결과 비교
#==============================
#to_categorical : (150, 3)
#get_dummies    : (150, 3)
#OneHotEncoder  : (150, 3)

#방법 1 == 방법 2 : True
#방법 1 == 방법 3 : True
#방법 2 == 방법 3 : True


#exit()

x_train, x_test, y_train, y_test = train_test_split(
                                x, y,
                                train_size=0.8,
                                random_state=333,
                                shuffle=True,
                                stratify=y,                       
)

print(x_train.shape, x_test.shape) # (120, 4) (30, 4)
print(y_train.shape, y_test.shape) # (120, 3) (30,3)

#exit()

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam

# Scaling
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성

#2. 모델구성
model = Sequential()
#model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, input_shape=(4,) activation='relu'))# 1차원 튜플(tuple) 입니다.
# Keras의 input_shape는 입력 데이터의 shape를 튜플 형태로 지정하므로,

model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))

model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))  # activation='softmax'는 출력층의 값을 확률 분포 형태로 변환하는 활성화 함수
                                           # 각 클래스에 속할 확률을 출력하고, 모든 클래스의 확률 합은 1이 됩니다.

# 원데이터 ------->>> input_shape
"""
(n, 4) ---------------(4,)
(n, 100, 3)------------->>>>>>>(100, 3)
(n, 100, 100, 3)-------->>>>>(100, 100, 3)
"""
#3. 컴파일  &  훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])  # sparse_categorical_crossentropy 정수 형태의 라벨이라면
es= EarlyStopping(                                                                        # 일반적인 loss='categorical_crossentropy'
    monitor= 'val_loss',
    mode= 'auto',
    patience=20,
    restore_best_weights=True,
)

start_time= time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=8,
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

y_test = np.argmax(y_test, axis=1) #axis=1은 **2차원 배열에서 "각 행(row)을 기준으로 계산하라"**는 의미입니다.
print(y_test)  # [0 2 0 1 1 1 0 2 0 2 2 2 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 1]



accuracy_score = accuracy_score(y_test, y_predict)
print('acc_score:', accuracy_score)
print('걸린시간:', round(end_time - start_time, 2), '초')



#loss : 0.2736741006374359
#acc: 0.9

#acc_score: 0.9
#걸린시간: 5.7 초