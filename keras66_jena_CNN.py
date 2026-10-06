#https://www.kaggle.com/datasets/mnassrib/jena-climate


# Jena Climate Dataset의 칼럼 개수는 총 15개입니다.

# (Date Time 포함, 또는 14개의 기상 특징 + 시간 칼럼으로 설명되기도 합니다.)
# 이 데이터셋은 Max Planck Institute for Biogeochemistry(Jena, Germany)에서 2009년 1월부터 2016년 12월까지 10분 간격으로 기록한 기상 시계열 데이터입니다.
# 칼럼 목록 및 한글 번역

# Jena Climate Dataset의 총 데이터(행) 개수는 420,551개입니다.
# 기간: 2009년 1월 1일 ~ 2016년 12월 31일 (일부 버전은 2017년 1월 1일 00:00까지 포함)
# 기록 간격: 10분마다
# 칼럼 수: 15개 (Date Time 포함)
# 이 숫자는 TensorFlow/Keras 공식 데이터셋(jena_climate_2009_2016.csv) 및 여러 튜토리얼·Kaggle 자료에서 공통적으로 확인되는 값입니다.
#############################################################################################################################################

# os.environ["TF_GPU_ALLOCATION"]= "cuda_malloc_async"  # 메모리 모으기

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.layers import Conv2D, Dense,Dropout, Flatten
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense
)
import time
from sklearn.metrics import accuracy_score
#########################################################################################

# #1. 데이터
# path = './_data/kaggle_jena/'
# dataset = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)

# print(dataset.shape)

# y_cor = datasets[-144:]['wd (deg)']
# print(y_cor.shape)
# ############################# 훈련 데이터 자르기  #################################
# x_data= datasets[:-288].drop(['wd(deg)'], axis=1)
# y_data= datasets[144:-144]['wd(deg)']

# print(x_data.shape)
# print(y_data.shape)

# size_x= 144
# size_y= 144

# def split_x(datasets, size):
#     aaa=[]
#     for i in range(len(dataset)- size + 1)
#         subset= dataset[i:i + size]
#         aaa.append(subset)
#     return np.array(aaa)

# start_time = time.time()
# x= split_x(x_data, size_x)
# y= split_x(x_data, size_y)
# end_time = time.time()


# print('x :', x.shape, 'y :', y.shape)
# print('자르는 시간:', round(end_time-start_time,2))

# #summit용 x 데이터
# x_predict = datasets[-288:-144].drop(['wd(deg)'])
# print(type(x_predict))

# x_predict= x_predict.to_numpy()
# print(x_predict.shape)
# x_predict= x_predict.reshape(1, 144, 13)

##################################################################################################################
# ============================================================
# 1. 데이터 로드
# ============================================================
path = './_data/kaggle_jena/'

df = pd.read_csv(path + "jena_climate_2009_2016.csv", index_col=0)
print(df.shape)          # (420551, 14)
print(df.columns)

# y = wd (deg)
y = df['wd (deg)'].values.reshape(-1, 1)

# x = wd를 제외한 나머지 특징
x = df.drop(['wd (deg)'], axis=1).values

print("x shape:", x.shape)   # (420551, 13)
print("y shape:", y.shape)   # (420551, 1)


# ============================================================
# 2. 스케일링
# ============================================================
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y)


# ============================================================
# 3. 시퀀스 생성 함수 (1시간 → 다음 1시간 예측)
# ============================================================
def split_xy_sequences(x, y, timesteps=6, predict_steps=6):
    """
    timesteps     : 입력으로 사용할 과거 길이 (1시간 = 6)
    predict_steps : 예측할 미래 길이 (1시간 = 6)
    """
    x_seq, y_seq = [], []
    
    for i in range(len(x) - timesteps - predict_steps + 1):
        x_seq.append(x[i : i + timesteps])
        y_seq.append(y[i + timesteps : i + timesteps + predict_steps].flatten())
    
    return np.array(x_seq), np.array(y_seq)


# ===== 여기가 핵심 변경 =====
TIMESTEPS = 6          # 과거 1시간 (10분 × 6)
PREDICT_STEPS = 6      # 다음 1시간 예측

x_seq, y_seq = split_xy_sequences(x_scaled, y_scaled, 
                                  timesteps=TIMESTEPS, 
                                  predict_steps=PREDICT_STEPS)

print("===== 시퀀스 생성 결과 =====")
print("x_seq shape:", x_seq.shape)   # (samples, 6, 13)
print("y_seq shape:", y_seq.shape)   # (samples, 6)


# ============================================================
# 4. 시간 순서 유지하며 데이터 분리
# ============================================================
train_size = int(len(x_seq) * 0.70)
val_size   = int(len(x_seq) * 0.15)

x_train = x_seq[:train_size]
y_train = y_seq[:train_size]

x_val   = x_seq[train_size:train_size + val_size]
y_val   = y_seq[train_size:train_size + val_size]

x_test  = x_seq[train_size + val_size:]
y_test  = y_seq[train_size + val_size:]

print("Train:", x_train.shape, y_train.shape)
print("Val  :", x_val.shape, y_val.shape)
print("Test :", x_test.shape, y_test.shape)

# Train: (294378, 6, 13) (294378, 6)
# Val  : (63081, 6, 13) (63081, 6)
# Test : (63081, 6, 13) (63081, 6)

# ============================================================
# 5. CNN 입력을 위한 4차원 변환
# ============================================================

# x_train = x_train.reshape(x_train.shape[0], 6, 13, 1)
# x_val   = x_val.reshape(x_val.shape[0], 6, 13, 1)
# x_test  = x_test.reshape(x_test.shape[0], 6, 13, 1)

# print("===== CNN 입력 형태 =====")
# print("x_train_cnn:", x_train.shape)
# print("x_val_cnn  :", x_val.shape)
# print("x_test_cnn :", x_test.shape)
# ===== CNN 입력 형태 =====
# x_train_cnn: (294378, 6, 13, 1)
# x_val_cnn  : (63081, 6, 13, 1)
# x_test_cnn : (63081, 6, 13, 1)

# ============================================================
# 2. LSTM MODEL
# ============================================================

model = Sequential()

model.add(
    LSTM(
        64,
        activation='relu',
        input_shape=(6, 13),
        return_sequences=True
    )
)

model.add(
    LSTM(
        32,
        activation='relu',
        return_sequences=True
    )
)

model.add(
    LSTM(
        32,
        activation='relu',
        return_sequences=True
    )
)

model.add(
    LSTM(
        16,
        activation='relu',
        return_sequences=True
    )
)

model.add(
    LSTM(
        32,
        activation='relu',
        return_sequences=False
    )
)

model.add(Dropout(0.2))

model.add(
    Dense(
        32,
        activation='relu'
    )
)

model.add(Dropout(0.2))

model.add(
    Dense(
        16,
        activation='relu'
    )
)

model.add(
    Dense(
        6
    )
)

model.summary()




from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #🤎🤎🤎🤎🤎

#3. 컴파일 훈련
model.compile(loss='mse',optimizer = 'adam')

es = EarlyStopping(monitor='val_loss', mode= 'min',
                    patience= 30,
                    restore_best_weights= True,
                    verbose=1, #🤎
                   )


#exit()
start_time = time.time()
hist = model.fit(x_train,y_train,
                 epochs=1000,batch_size=32,validation_split = 0.2,
                callbacks=[es,],      
                # callbacks=[es,mcp],           #🤎🤎🤎🤎🤎
                verbose=1,
                 )
end_time = time.time()


print("=======================================")
#4. 평가예측 
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)
results = model.predict(x_test)
# print(results)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score , mean_squared_error
r2 =  r2_score(y_test, y_predict)
print ("r2 = :" , r2)

#loss(mse) :34.15455627441406
# r2 = : 0.589704701049621 (0.75이상 )

mse = mean_squared_error(y_test, y_predict) #원값 과 예측값
print("mse :" , mse)

def RMSE(y_test, y_predict):         # RMSE 함수를 정의하기
    return np.sqrt(mean_squared_error(y_test, y_predict))
    
rmse  = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

# # ============================================================
# # 2. Model Construction - DNN
# # ============================================================

# model = Sequential()

# # 28×28×1 image → 784-dimensional vector
# model.add(Flatten(input_shape=(6, 13, 1)))

# # Fully Connected Layer
# model.add(Dense(128, activation='relu'))
# model.add(BatchNormalization())
# # Dropout for regularization
# model.add(Dropout(0.2))

# # Fully Connected Layer
# model.add(Dense(64, activation='relu'))
# model.add(BatchNormalization())
# # Dropout for regularization
# model.add(Dropout(0.2))

# # Fully Connected Layer
# model.add(Dense(32, activation='relu'))
# model.add(BatchNormalization())
# # Dropout for regularization
# model.add(Dropout(0.2))

# # Fully Connected Layer
# model.add(Dense(16, activation='relu'))
# model.add(BatchNormalization())
# # Output Layer - 10 classes
# model.add(Dense(6, activation='softmax'))

# # Model Summary
# model.summary()

# #3. 컴파일 , 훈련
# model.compile(loss='categorical_crossentropy', optimizer = 'adam',
#             metrics = ['acc'],
#             )   

# start_time = time.time()
# model.fit(x_train,y_train, epochs=10, batch_size=32,
#         verbose = 1, 
#         validation_split = 0.2,
#           )
# end_time = time.time()


# #4.평가, 예측
# print("===============================model.evaluate================================")
# loss = model.evaluate(x_test, y_test, verbose=1)
# print('loss:', loss[0] )
# print('acc:', loss[1])

# y_predict = model.predict(x_test)

# y_predict = np.argmax(y_predict,axis=1,)#.reshape(-1,1)
# y_test = np.argmax(y_test,axis=1,)#.reshape(-1,1)

# acc_score = accuracy_score(y_test, y_predict)
# print('accuracy_score : ', acc_score)
# print('걸린시간 : ', round(end_time-start_time,2), '초')