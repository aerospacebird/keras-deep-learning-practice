# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data

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



print(train_csv.columns) 

print(train_csv.info())
print(test_csv.info())

print(train_csv.describe()) 

################################### 결측치 확인 ###########################################################################
print(train_csv.isna().sum())
print(test_csv.isnull().sum())

###################################### x, y 분리 ###########################################################################

x = train_csv.drop(['target'], axis=1)
print(x)   #

y = train_csv['target']
print(y)    
print(x.shape, y.shape)    #(200000, 200) (200000,)

print(np.unique(y, return_counts=True))
#(array([0, 1]), array([179902, 20098]))


"""
x_train, x_test, y_train, y_test = train_test_split(
                                   x, y,
    
                                   random_state=42,
)
"""
####################################### submit 물밑작업 #######################################################

print(test_csv.info())

###################################결측치 처리 2.  평균값 넣기     #################################################
#test_csv = test_csv.fillna(test_csv.mean())
#print(test_csv.info())  # (6493, 8)
#print(test_csv.shape)  # (6493, 8)
x_train, x_test, y_train, y_test = train_test_split(x, y,
                 train_size=0.75,
                 #test_size=0.25,
                 shuffle=True,   # 디폴트 섞는다. 75%/25% 비율로 섞는다.
                 random_state=337, # # 42  디폴트 값이다. 이값을 변경해도 성능에 영향을 줄수가 있다.
                 stratify=y #훈련 데이터와 테스트 데이터를 나눌 때, y의 클래스 비율이 원본 데이터와 최대한 동일하게 유지되도록 분할하는 옵션입니다.               
)
print(x_train.shape, x_test.shape) # (398, 30)  (171, 30)
print(y_train.shape, y_test.shape) # (398, )    (171, )
#############################################################################################
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
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
#0.0 1.0000000000000002
#-0.07306255183145582 1.0796097666145639
#exit()
#print(np.unique(y_train, return_counts=True))
#print(np.unique(y_test, return_counts=True))


#2. 모델구성
"""
model = Sequential()
model.add(Dense(30, input_dim=200, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(240, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 시그모이드 함수는 입력된 모든 실수를 0과 1 사이의 값으로 변환하여 부드러운 
                                          #  S자 형태(Sigmoid curve)로 출력하는 수학 함수입니다.
"""
#########################################################################################################################
#2-2  함수형 모델 구성

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# 2. 함수형 모델 구성
input1 = Input(shape=(200,), name='input1')

dense1 = Dense(30, name='ys1')(input1)
dense2 = Dense(120, name='ys2')(dense1)
dense3 = Dense(240, name='ys3')(dense2)
dense4 = Dense(120, name='ys4')(dense3)
dense5 = Dense(60, name='ys5')(dense4)
dense6 = Dense(30, name='ys6')(dense5)
dense7 = Dense(1, name='ys7')(dense6)

output1 = Dense(1, name='output1')(dense7)

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
    filepath=path + 'keras32_mcp7.keras',
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

model = load_model('./_save/keras32/keras32_mcp7.keras')

print("모델 로드 완료!")
model.summary()





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


#loss: 0.23901839554309845  # StandardScaler
#acc: 0.9123                # StandardScaler


####################### submission.csv 만들기 // 칼럼에 값을 넣어준다.##########################################################
print(submission)
y_submit = model.predict(test_csv)



submission['target'] = y_submit
print(submission)
print(submission.shape)

submission.to_csv(path + "submit/" + "submit_0908_1632.csv")

"""
print("########################################################## history #########################################################")
print(hist)
print("########################################################## history #########################################################")
print(hist.history)
print("########################################################## loss #########################################################")
print(hist.history['loss'])

print("######################################################### val_loss #########################################################")
print(hist.history['val_loss'])

print("#############################################################################################################################")

      
# 결과는 Dictionary형태로 나온다.

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 글꼴 깨짐현상 수정
plt.rcParams['axes.unicode_minus'] = False     # 한글 글꼴 깨짐현상 수정

plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'][3:], c='red', label= 'loss') # y값만 넣으면 시간순으로 그려줌
plt.plot(hist.history['val_loss'][3:], c='blue', label= 'val_loss')
#plt.xlim()
#plt.ylim()
plt.legend(loc='upper right')
plt.title('kaggle_santander')  # 한글 글꼴 깨짐현상 수정
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()  # 격자 표시 추가
plt.show()
"""

# ============================================================
# History 확인
# ============================================================

print("=" * 70)
print("History Object")
print("=" * 70)
print(hist)

print("\n" + "=" * 70)
print("History Dictionary")
print("=" * 70)
print(hist.history)

print("\n" + "=" * 70)
print("Loss")
print("=" * 70)
print(hist.history['loss'])

print("\n" + "=" * 70)
print("Validation Loss")
print("=" * 70)
print(hist.history['val_loss'])


# ============================================================
# Loss / Validation Loss 시각화
# ============================================================

import matplotlib.pyplot as plt

# 한글 깨짐 방지
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# History에서 데이터 추출
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# Epoch 번호
epochs = range(1, len(loss) + 1)

# 그래프
plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    loss,
    label='Training Loss',
    linewidth=2
)

plt.plot(
    epochs,
    val_loss,
    label='Validation Loss',
    linewidth=2
)

plt.title('Kaggle Santander - Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()