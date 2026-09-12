# Classification


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score #
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


#print(x.shape, y.shape)  #(1797, 64) (1797,)
#print(y)
#print(np.unique(y, return_counts=True))  # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]))


#1. 데이터
path = './_data/kaggle_santander/'  # 상대경로

#path = "c:/study/_data/kaggle_santander/"  # 절대경로
#path = "c:/study/_data/ddarung/"  # 슬래시 역슬래시 상관없어
#path = "c://study//_data//ddarung/"  # 슬래시 2개도 가능하다.  가독성있게 작성하라.

train_csv = pd.read_csv(path + "train.csv", index_col=0) #1459 rows x 11columns ------->>>>#1459 rows x 10 columns index_col=0 때문이다.
print(train_csv.shape) #[200000 rows x 200 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv) #[200000 rows x 200 columns]

submission = pd.read_csv(path + "sample_submission.csv", index_col=0)
print(submission) #[6493 rows x 1 columns]

print(train_csv.shape) #[200000 rows x 200 columns]
print(test_csv.shape) # [200000 rows x 200 columns]
print(submission.shape) # [200000 rows x 200 columns]

#exit()

print(train_csv.columns) 

print(train_csv.info())
print(test_csv.info())

print(train_csv.describe()) 

###################################### x, y 분리 ###########################################################################
x = train_csv.drop(['target'], axis=1)
print(x)   #

y = train_csv['target']
print(y)    
print(x.shape, y.shape)    #(200000, 200) (200000,)

print(np.unique(y, return_counts=True))
#(array([0, 1]), array([179902, 20098]))

# 판다스를 넘파이로 바꾸기
#y = np.array(y)
#y = y.to_numpy()


print(x.shape, y.shape)  #(1797, 64) (1797,)
print(y)
print(np.unique(y, return_counts=True))  # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]))
#[200000 rows x 1 columns]
#(200000, 201)
#(200000, 200)
#(200000, 1)
# 
# 
# 
# 
# exit()
################################### 결측치 확인 ###########################################################################
print(train_csv.isna().sum())
print(test_csv.isnull().sum())

###################################### x, y 분리 ###########################################################################
"""
x = train_csv.drop(['target'], axis=1)
print(x)   #

y = train_csv['target']
print(y)    
print(x.shape, y.shape)    #(200000, 200) (200000,)

print(np.unique(y, return_counts=True))
#(array([0, 1]), array([179902, 20098]))
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

# 판다스를 넘파이로 바꾸기
#y = np.array(y)
#y = y.to_numpy()
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
#exit()

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam

# 핵심 목적은 x_train과 x_test의 각 특성(feature)을 비슷한 스케일로 맞춰서 머신러닝/딥러닝 모델이 안정적으로 학습하도록 하는 것

# Scaling
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)   #Train과 Test가 서로 다른 기준으로 Scaling됩니다.훈련 데이터의 평균과 표준편차를 계산
x_test = scaler.transform(x_test)    # 계산한 평균과 표준편차로 데이터를 변환

#2. 모델구성
model = Sequential()
model.add(Dense(256, input_dim=200, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))  # activation='softmax'는 출력층의 값을 확률 분포 형태로 변환하는 활성화 함수
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
model.fit(x_train, y_train, epochs=900, batch_size=64,
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


#loss : 0.24126386642456055
#acc: 0.91

#acc_score: 0.911425
#걸린시간: 220.79 초