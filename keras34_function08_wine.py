from sklearn.datasets import load_wine
# acc = 0.95 이상을 만들어 보세요?

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
 
from tensorflow.keras.utils import to_categorical



#1. DATA준비

datasets = load_wine()    #머신러닝의 다중분류(Multi-class Classification)

print(datasets)
print(datasets.data)
print(datasets.DESCR)
print(datasets.feature_names)

#:Missing Attribute Values: None
#:Class Distribution: class_0 (59), class_1 (71), class_2 (48)

x = datasets.data
y = datasets['target']

print(x.shape, y.shape)  #(178, 13) (178,)
print(y)
print(np.unique(y, return_counts=True))  # (array([0, 1, 2]), array([59, 71, 48]))


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
print(y.shape)  #(178, 3)    one hot encoding한후에 split 시켜라!!!  아니면 2번 연속 반복해야 한다.
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
###########################################################################################
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = MaxAbsScaler()
scaler = RobustScaler()


scaler.fit(x_train)  #준비
x_train = scaler.transform(x_train)  #변환

x_train = scaler.fit_transform(x_train)  #변환
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

#0.0 1.0000000000000002
#-0.04595404595404595 1.1248751248751248
#exit()
###########################################################################################
"""
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam

# Scaling
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


"""
#2. 모델구성
"""
model = Sequential()
model.add(Dense(32, input_dim=13, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))  # activation='softmax'는 출력층의 값을 확률 분포 형태로 변환하는 활성화 함수
 """                                          # 각 클래스에 속할 확률을 출력하고, 모든 클래스의 확률 합은 1이 됩니다.
######################################################################################################################

#2-2  함수형 모델 구성

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# 2. 함수형 모델 구성
input1 = Input(shape=(13,), name='input1')

dense1 = Dense(32, name='ys1')(input1)
dense2 = Dense(16, name='ys2')(dense1)
dense3 = Dense(8, name='ys3')(dense2)
dense4 = Dense(3, name='ys4')(dense3)

output1 = Dense(1, name='output1')(dense4)

model2 = Model(
    inputs=input1,
    outputs=output1,
    name='Functional_Model'
)

model2.summary()

exit()










from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor= 'val_loss', mode='min',
                   patience=20,
                   restore_best_weights=True,
                   verbose=1
                   )
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
    filepath=path + 'keras32_mcp8.keras',
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

model = load_model('./_save/keras32/keras32_mcp8.keras')

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



#loss : 0.015163246542215347
#acc: 1.0

#acc_score: 1.0
#걸린시간: 60.45 초

#loss : 0.00042849453166127205
#acc: 1.0

#acc_score: 1.0
#걸린시간: 59.76 초

#acc_score: 1.0     # StandardScaler
#걸린시간: 58.98 초  # StandardScaler

#loss : 0.03230027109384537   # MaxAbsScaler
#acc: 1.0

#acc_score: 1.0               # MaxAbsScaler
#걸린시간: 15.18 초