# Classification


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_breast_cancer  #유방암관련 데이터셋을 불러오기

#1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)
print(datasets.feature_names)

#x = datasets.data
x = datasets['data']
y = datasets.target

print(x.shape, y.shape)    #(569, 30) (569,)
print(type(x))  # <class 'numpy.ndarray'>
print(y)
# 0과 1의 갯수를 몇개인지 찾아보기._numpy

print(pd.DataFrame(y).value_counts())

"""
1    357
0    212
Name: count, dtype: int64
[0 1]
(array([0, 1]), array([212, 357]))

"""

print(pd.Series(y).value_counts)
"""
1    357
0    212
Name: count, dtype: int64
"""
print(np.unique(y))  # [0 1]    unique() # 분류 클라스의 종류와 갯수를 파악하는데  상당히 좋은 함수이다.
print(np.unique(y, return_counts=True))  # (array([0, 1]), array([212, 357]))

"""  #  block comments
#x_train, x_test, y_train, y_test = train_test_split(x, y,
                 train_size=0.75,
                 #test_size=0.25,
                 shuffle=True,   # 디폴트 섞는다. 75%/25% 비율로 섞는다.
                 random_state=500,   # 42  디폴트 값이다. 이값을 변경해도 성능에 영향을 줄수가 있다.                
)


""" # block comments
x_train, x_test, y_train, y_test = train_test_split(x, y,
                 train_size=0.75,
                 #test_size=0.25,
                 shuffle=True,   # 디폴트 섞는다. 75%/25% 비율로 섞는다.
                 random_state=337, # # 42  디폴트 값이다. 이값을 변경해도 성능에 영향을 줄수가 있다.
                 stratify=y #훈련 데이터와 테스트 데이터를 나눌 때, y의 클래스 비율이 원본 데이터와 최대한 동일하게 유지되도록 분할하는 옵션입니다.               
)
print(x_train.shape, x_test.shape) # (398, 30)  (171, 30)
print(y_train.shape, y_test.shape) # (398, )    (171, )


#print(np.unique(y_train, return_counts=True))
#print(np.unique(y_test, return_counts=True))


#2. 모델구성
model = Sequential()
model.add(Dense(30, input_dim=30, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(240, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 시그모이드 함수는 입력된 모든 실수를 0과 1 사이의 값으로 변환하여 부드러운 
                                          #  S자 형태(Sigmoid curve)로 출력하는 수학 함수입니다.

#3. compile, training
model.compile(
    loss='binary_crossentropy',  # 이진분류에 해당되며,  loss함수가 binary_crossentropy
    optimizer='adam',
    metrics=['acc']
)

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min', 
    patience=100,  # monitor='val_loss'라면:   val_loss가 최저값을 갱신하지 않는 epoch가 10번 연속 발생하면 학습을 중단한다.
    restore_best_weights=True,

             )



model.fit(x_train, y_train, epochs=1000, batch_size=32,
          verbose=1,
          callbacks=[es],
          validation_split=0.3,
          )
end_time = time.time()


#4. 평가 및 추론

loss = model.evaluate(x_test, y_test)
print('loss:', loss[0])
print('acc:', round(loss[1], 4))
print("#######################################################################")

y_pred = model.predict(x_test)
print('y_pred:', y_pred)
#print(y_pred[:10])
#result = model.predict()
y_pred = np.round(y_pred) # 0, 1의 결과를 추출하기 위해서, ROUND함수를 사용한다.
print(y_pred[:10])

from sklearn.metrics import accuracy_score
acc_score= accuracy_score(y_test, y_pred)
print("acc_score :", acc_score)



