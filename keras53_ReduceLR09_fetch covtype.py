from sklearn.datasets import fetch_covtype

# acc = 0.93 이상을 넘겨라!!

from sklearn.datasets import load_wine
# acc = 0.95 이상을 만들어 보세요?

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

datasets = fetch_covtype()    #머신러닝의 다중분류(Multi-class Classification)

print(datasets)
print(datasets.data)
print(datasets.DESCR)
print(datasets.feature_names)

#**Data Set Characteristics:**

#=================   ============
#Classes                        7
#Samples total             581012
#Dimensionality                54
#Features                     int
#=================   ============
#exit()
x = datasets.data
y = datasets['target']


print(x.shape, y.shape)  #shape=(581012, 54)) (581012,)
print(y)
print(np.unique(y, return_counts=True))  # (array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))

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

y = datasets['target']

# Covtype의 target은 1~7이므로 0~6으로 변환
y = y - 1  #  (581012, 8)-------- >>>>>>>>>>>>(581012, 7)

# One-Hot Encoding
y = to_categorical(y, num_classes=7)

print(x.shape, y.shape)
print(y[:10])


#y = to_categorical(y)
print(y)
print(y.shape)  #(581012, 8)-------- >>>>>>>>>>>>(581012, 7)   one hot encoding한후에 split 시켜라!!!  아니면 2번 연속 반복해야 한다.


# One-Hot Encoding을 하는 가장 중요한 이유는 범주형 데이터(Category)를 인공지능 모델이 이해할 수 있는 숫자 형태로 변환하기 위해서입니다.
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

print(x_train.shape, x_test.shape) # (464809, 54) (116203, 54)
print(y_train.shape, y_test.shape) # (464809, 54) (116203, 54)

#exit()
################################################################################################################
from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler
#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = MaxAbsScaler()
scaler = RobustScaler()

#scaler.fit(x_train)  #준비
#x_train = scaler.transform(x_train)  #변환

x_train = scaler.fit_transform(x_train)  #변환
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

#0.0 1.0
#0.0 1.0050359712230217
#exit()
###################################################################################################################
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam

# Scaling
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=54, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(7, activation='softmax'))  # activation='softmax'는 출력층의 값을 확률 분포 형태로 변환하는 활성화 함수
                                           # 각 클래스에 속할 확률을 출력하고, 모든 클래스의 확률 합은 1이 됩니다.

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#3. 컴파일, 훈련

from tensorflow.keras.optimizers import Adam
learning_rate = 0.001

# learnig_rate = 0.001
# learnig_rate = 0.0001
# learnig_rate = 0.005
# learnig_rate = 0.05
# learnig_rate = 0.009


#model.compile(loss='mse', optimizer= Adam(learning_rate= learning_rate))


# model.compile(
#     loss='binary_crossentropy',  #  loss함수가 binary_crossentropy
#     optimizer= Adam(learning_rate= learning_rate),
#     metrics=['acc']
# )
model.compile(loss='categorical_crossentropy', optimizer= Adam(learning_rate= learning_rate), metrics=['acc'])  # sparse_categorical_crossentropy 정수 형태의 라벨이라면
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=40,
    verbose=1,
    restore_best_weights=True,
)
rlr=ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5,
)


# model.compile(loss='mse', optimizer='adam')
# es = EarlyStopping(monitor= 'val_loss', mode='min',
#                    patience=20,
#                    restore_best_weights=True,
#                    verbose=1
#                    )
################################################# mcp 세이브 파일명 만들기 ##################################################
import datetime
date = datetime.datetime.now()  # 2026-09-14 11:41:15.799024

print(date)
print(type(date))  #  <class 'datetime.datetime'>

date = date.strftime("%m%d_%H%M")
print(date)
print(type(date))

#2026-09-14 11:48:29.645739
#<class 'datetime.datetime'>
#0914_1148
#<class 'str'>

path= './_save/keras32/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filename = "".join([path, "k32_", date, "-" ,filename])

# 내가 생각하는 파일명
# './save/keras30'+ "k30_"+ "0914_1147"+ "530-0.001.keras"


#exit()







mcp= ModelCheckpoint(
    monitor='val_loss',
    mode= 'auto',
    save_best_only=True,
    filepath=path + 'keras32_mcp9.keras',
    verbose=1
      # loss도 상관없다.
)

start_time = time.time()  #시작시간 반환
hist = model.fit(x_train, y_train, epochs=1000, batch_size=4, 
          verbose=1,
          #validation_data = (x_val, y_val),
          callbacks= [es,mcp],
    
          #verbose=1
          )  # verbose=0 훈련과정을 보여주지 않고 결과만 보여준다. 
end_time = time.time()  # 끝 시간 반환
# verbose=1은 Keras/TensorFlow에서 모델 학습 과정의 진행 상황을 화면에 출력하라는 설정입니다.  verbose=1 default value, 
# verbose=0 : 침묵(아무것도 시현이 않됨)
# verbose=2 : 프로그래스바 삭제,
# verbose=3 : epochs만 나옴,
# verbose= 나머지 : epochs만 나옴.


#model.save(path + 'keras29_3_save_model.keras')   #  훈련이 끝난 상태로 저장하여, 성능이 어느정도 보장된 상태로 저장하였다.
#model.save_weights(path + 'keras29_5_save_weights2.h5')   #  훈련이 끝난 상태로 저장하여, 성능이 어느정도 보장된 상태로 저장하였다.
#model.save_weights(path + 'keras29_5_save_weights2.h5')   #  훈련이 끝난 상태로 저장하여, 성능이 어느정도 보장된 상태로 저장하였다.

#exit()
from tensorflow.keras.models import load_model

model = load_model('./_save/keras32/keras32_mcp9.keras')

print("모델 로드 완료!")
model.summary()




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



#loss : 0.25864943861961365
#acc: 0.9

#acc_score: 0.8989268779635637
# 걸린시간: 960.52 초

#loss : 0.2514115869998932
#acc: 0.9

#acc_score: 0.9016806795005292
#걸린시간: 897.13 초


#loss : 0.25711965560913086   #  StandardScaler
#acc: 0.9

#acc_score: 0.8976102166037021 #  StandardScaler
#걸린시간: 547.23 초

#loss : 0.2527783513069153   #MaxAbsScaler
#acc: 0.9

#acc_score: 0.8994087932325328  #MaxAbsScaler
#걸린시간: 1703.17 초