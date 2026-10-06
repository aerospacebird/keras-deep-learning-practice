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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
#########################################################################################
# ============================================================
# 1. 라이브러리
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)


# ============================================================
# 2. 데이터 로드
# ============================================================

path = './_data/kaggle_jena/'

df = pd.read_csv(
    path + "jena_climate_2009_2016.csv",
    index_col=0
)

print("전체 데이터:", df.shape)
print(df.columns)


# ============================================================
# 3. X / y 분리
# ============================================================

# y = 풍향 wd (deg)
y = df['wd (deg)'].values.reshape(-1, 1)

# x = wd를 제외한 나머지 13개 특징
x = df.drop(['wd (deg)'], axis=1).values

print("\n===== 원본 데이터 =====")
print("x shape:", x.shape)
print("y shape:", y.shape)


# ============================================================
# 4. 스케일링
# ============================================================

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y)

print("\n===== Scaling 완료 =====")
print("x_scaled:", x_scaled.shape)
print("y_scaled:", y_scaled.shape)


# ============================================================
# 5. 시퀀스 생성
# ============================================================

def split_xy_sequences(
    x,
    y,
    timesteps=6,
    predict_steps=6
):
    """
    timesteps     : 과거 입력 길이
                    10분 × 6 = 1시간

    predict_steps : 미래 예측 길이
                    10분 × 6 = 1시간
    """

    x_seq = []
    y_seq = []

    for i in range(
        len(x) - timesteps - predict_steps + 1
    ):

        # 과거 1시간
        x_seq.append(
            x[i:i + timesteps]
        )

        # 미래 1시간
        y_seq.append(
            y[
                i + timesteps:
                i + timesteps + predict_steps
            ].flatten()
        )

    return np.array(x_seq), np.array(y_seq)


# ============================================================
# 6. 시간 설정
# ============================================================

TIMESTEPS = 6
PREDICT_STEPS = 6


x_seq, y_seq = split_xy_sequences(
    x_scaled,
    y_scaled,
    timesteps=TIMESTEPS,
    predict_steps=PREDICT_STEPS
)


print("\n===== 시퀀스 생성 결과 =====")
print("x_seq shape:", x_seq.shape)
print("y_seq shape:", y_seq.shape)
# ===== 시퀀스 생성 결과 =====
# x_seq shape: (420540, 6, 13)
# y_seq shape: (420540, 6)
# 예상
# x_seq = (samples, 6, 13)
# y_seq = (samples, 6)


# ============================================================
# 7. 시간 순서를 유지한 데이터 분리
# ============================================================

train_size = int(len(x_seq) * 0.70)
val_size   = int(len(x_seq) * 0.15)

x_train = x_seq[:train_size]
y_train = y_seq[:train_size]

x_val = x_seq[
    train_size:
    train_size + val_size
]

y_val = y_seq[
    train_size:
    train_size + val_size
]

x_test = x_seq[
    train_size + val_size:
]

y_test = y_seq[
    train_size + val_size:
]


print("\n===== 데이터 분할 =====")
print("Train:", x_train.shape, y_train.shape)
print("Val  :", x_val.shape, y_val.shape)
print("Test :", x_test.shape, y_test.shape)
# ===== 데이터 분할 =====
# Train: (294378, 6, 13) (294378, 6)
# Val  : (63081, 6, 13) (63081, 6)
# Test : (63081, 6, 13) (63081, 6)

#exit()
# ============================================================
# 8. Conv1D 모델
# ============================================================

model = Sequential([

    # --------------------------------------------------------
    # 입력
    # (batch, timesteps, features)
    # (batch, 6, 13)
    # --------------------------------------------------------

    Conv1D(
        filters=64,
        kernel_size=3,
        padding='same',
        activation='relu',
        input_shape=(
            TIMESTEPS,
            x_train.shape[2]
        )
    ),

    Dropout(0.2),


    # --------------------------------------------------------
    # 두 번째 Conv1D
    # --------------------------------------------------------

    Conv1D(
        filters=128,
        kernel_size=3,
        padding='same',
        activation='relu'
    ),

    Dropout(0.2),


    # --------------------------------------------------------
    # 세 번째 Conv1D
    # --------------------------------------------------------

    Conv1D(
        filters=64,
        kernel_size=3,
        padding='same',
        activation='relu'
    ),


    # --------------------------------------------------------
    # 시간축을 하나의 벡터로 압축
    # --------------------------------------------------------

    GlobalAveragePooling1D(),


    # --------------------------------------------------------
    # Dense
    # --------------------------------------------------------

    Dense(
        64,
        activation='relu'
    ),

    Dropout(0.2),

    Dense(
        32,
        activation='relu'
    ),


    # --------------------------------------------------------
    # 미래 6개 시점 예측
    # --------------------------------------------------------

    Dense(
        PREDICT_STEPS
    )
])


# ============================================================
# 9. Compile
# ============================================================

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)


# ============================================================
# 10. 모델 구조
# ============================================================

model.summary()


# ============================================================
# 11. Callback
# ============================================================

es = EarlyStopping(
    monitor='val_loss',
    patience=12,
    restore_best_weights=True,
    verbose=1
)


rlr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=6,
    min_lr=1e-6,
    verbose=1
)


# ============================================================
# 12. 학습
# ============================================================

history = model.fit(

    x_train,
    y_train,

    validation_data=(
        x_val,
        y_val
    ),

    epochs=100,

    batch_size=64,

    callbacks=[
        es,
        rlr
    ],

    verbose=1
)


# ============================================================
# 13. Test 예측
# ============================================================

y_pred_scaled = model.predict(
    x_test,
    verbose=0
)


print("\nPrediction shape:")
print(y_pred_scaled.shape)

# (samples, 6)


# ============================================================
# 14. 원래 단위(deg)로 복원
# ============================================================

y_test_orig = scaler_y.inverse_transform(
    y_test.reshape(-1, 1)
).reshape(y_test.shape)


y_pred_orig = scaler_y.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
).reshape(y_pred_scaled.shape)


print("\n===== 복원 후 =====")
print("y_test_orig:", y_test_orig.shape)
print("y_pred_orig:", y_pred_orig.shape)


# ============================================================
# 15. 1차원으로 펼치기
# ============================================================

y_true_flat = y_test_orig.flatten()
y_pred_flat = y_pred_orig.flatten()


# ============================================================
# 16. MSE
# ============================================================

mse = mean_squared_error(
    y_true_flat,
    y_pred_flat
)


# ============================================================
# 17. MAE
# ============================================================

mae = mean_absolute_error(
    y_true_flat,
    y_pred_flat
)


# ============================================================
# 18. RMSE
# ============================================================

rmse = np.sqrt(mse)


# ============================================================
# 19. R²
# ============================================================

r2 = r2_score(
    y_true_flat,
    y_pred_flat
)


# ============================================================
# 20. ±5도 정확도
# ============================================================

accuracy_5 = np.mean(
    np.abs(
        y_true_flat - y_pred_flat
    ) <= 5
) * 100


# ============================================================
# 21. ±10도 정확도
# ============================================================

accuracy_10 = np.mean(
    np.abs(
        y_true_flat - y_pred_flat
    ) <= 10
) * 100


# ============================================================
# 22. 최종 평가
# ============================================================

print("\n")
print("=" * 60)
print("          Conv1D Test Performance")
print("=" * 60)

print(f"MSE       : {mse:.4f}")
print(f"MAE       : {mae:.4f}")
print(f"RMSE      : {rmse:.4f}")
print(f"R²        : {r2:.4f}")
print(f"±5° 정확도  : {accuracy_5:.2f}%")
print(f"±10° 정확도 : {accuracy_10:.2f}%")

print("=" * 60)


# ============================================================
# 23. 마지막 시퀀스를 이용한 최종 1시간 예측
# ============================================================

last_sequence = x_seq[-1:]

print("\n===== 마지막 입력 시퀀스 =====")
print("shape:", last_sequence.shape)


y_final_pred_scaled = model.predict(
    last_sequence,
    verbose=0
)


# 원래 풍향(deg)으로 복원
y_final_pred = scaler_y.inverse_transform(
    y_final_pred_scaled.reshape(-1, 1)
).reshape(
    y_final_pred_scaled.shape
)


# ============================================================
# 24. 최종 예측 결과
# ============================================================

print("\n")
print("=" * 60)
print("       최종 1시간 풍향 예측")
print("=" * 60)

for i, value in enumerate(
    y_final_pred[0],
    start=1
):

    print(
        f"{i}번째 10분 후 : {value:.2f}°"
    )

print("=" * 60)


# ============================================================
# 25. 학습 곡선 - Loss
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    history.history['loss'],
    label='Train Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title(
    'Conv1D Training / Validation Loss'
)

plt.xlabel('Epoch')
plt.ylabel('MSE Loss')

plt.legend()
plt.grid()

plt.show()


# ============================================================
# 26. 학습 곡선 - MAE
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    history.history['mae'],
    label='Train MAE'
)

plt.plot(
    history.history['val_mae'],
    label='Validation MAE'
)

plt.title(
    'Conv1D Training / Validation MAE'
)

plt.xlabel('Epoch')
plt.ylabel('MAE')

plt.legend()
plt.grid()

plt.show()
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

# train_test_split
#scaling
#################################################################################################
# # Jena 데이터를 144개씩 자른 x, y 상태로 npy 저장
# import pandas as pd
# import numpy as np
# import time

# #1.데이터
# path = './_data/kaggle_jena/'
# datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
# print(datasets.shape)  # (420551, 14)

# y_cor = datasets[-144:]['wd (deg)'].to_numpy(dtype=np.float32)

# # 배열로 바꾼 뒤 144개씩 자르기. float32로 저장할 메모리 크기를 줄임.
# x_data = datasets[:-288].drop(['wd (deg)'],axis=1).to_numpy(dtype=np.float32)
# y_data = datasets[144:-144]['wd (deg)'].to_numpy(dtype=np.float32)

# print(x_data.shape)  # (420263, 13)
# print(y_data.shape)  # (420263,)

# size_x = 144
# size_y = 144

# def split_x(dataset, size):
#     aaa = []
#     for i in range(len(dataset) - size + 1):
#         subset = dataset[i:i + size]
#         aaa.append(subset)
#     return np.array(aaa)

# start_time = time.time()
# x = split_x(x_data, size_x)
# y = split_x(y_data, size_y)
# end_time = time.time()

# print('x :', x.shape, 'y :', y.shape)
# print('자르는 시간:', round(end_time-start_time,2))

# # 마지막 정답 144개 바로 앞의 입력 144개도 스케일링 전 상태로 저장.
# x_predict = datasets[-288:-144].drop(['wd (deg)'],axis=1).to_numpy(dtype=np.float32)
# x_predict = x_predict.reshape(1,144,13)

# print('x_predict:', x_predict.shape, 'y_cor:', y_cor.shape)

# #2.npy 저장
# np_path = './_data/kaggle_jena_npy/'
# np.save(np_path + 'keras58_01_x.npy', arr=x)
# np.save(np_path + 'keras58_01_y.npy', arr=y)
# np.save(np_path + 'keras58_01_x_predict.npy', arr=x_predict)
# np.save(np_path + 'keras58_01_y_cor.npy', arr=y_cor)
# print('npy 저장 완료')
# ##################################################################################################
# # ============================================================
# # 1. 데이터 로드
# # ============================================================
# path = './_data/kaggle_jena/'

# df = pd.read_csv(path + "jena_climate_2009_2016.csv", index_col=0)
# print(df.shape)          # (420551, 14)
# print(df.columns)

# # y = wd (deg)
# y = df['wd (deg)'].values.reshape(-1, 1)

# # x = wd를 제외한 나머지 특징
# x = df.drop(['wd (deg)'], axis=1).values

# print("x shape:", x.shape)   # (420551, 13)
# print("y shape:", y.shape)   # (420551, 1)


# # ============================================================
# # 2. 스케일링
# # ============================================================
# scaler_x = StandardScaler()
# scaler_y = StandardScaler()

# x_scaled = scaler_x.fit_transform(x)
# y_scaled = scaler_y.fit_transform(y)


# # ============================================================
# # 3. 시퀀스 생성 함수 (1시간 → 다음 1시간 예측)
# # ============================================================
# def split_xy_sequences(x, y, timesteps=6, predict_steps=6):
#     """
#     timesteps     : 입력으로 사용할 과거 길이 (1시간 = 6)
#     predict_steps : 예측할 미래 길이 (1시간 = 6)
#     """
#     x_seq, y_seq = [], []
    
#     for i in range(len(x) - timesteps - predict_steps + 1):
#         x_seq.append(x[i : i + timesteps])
#         y_seq.append(y[i + timesteps : i + timesteps + predict_steps].flatten())
    
#     return np.array(x_seq), np.array(y_seq)


# # ===== 여기가 핵심 변경 =====
# TIMESTEPS = 6          # 과거 1시간 (10분 × 6)
# PREDICT_STEPS = 6      # 다음 1시간 예측

# x_seq, y_seq = split_xy_sequences(x_scaled, y_scaled, 
#                                   timesteps=TIMESTEPS, 
#                                   predict_steps=PREDICT_STEPS)

# print("===== 시퀀스 생성 결과 =====")
# print("x_seq shape:", x_seq.shape)   # (samples, 6, 13)
# print("y_seq shape:", y_seq.shape)   # (samples, 6)


# # ============================================================
# # 4. 시간 순서 유지하며 데이터 분리
# # ============================================================
# train_size = int(len(x_seq) * 0.70)
# val_size   = int(len(x_seq) * 0.15)

# x_train = x_seq[:train_size]
# y_train = y_seq[:train_size]

# x_val   = x_seq[train_size:train_size + val_size]
# y_val   = y_seq[train_size:train_size + val_size]

# x_test  = x_seq[train_size + val_size:]
# y_test  = y_seq[train_size + val_size:]

# print("Train:", x_train.shape, y_train.shape)
# print("Val  :", x_val.shape, y_val.shape)
# print("Test :", x_test.shape, y_test.shape)


# # ============================================================
# # 5. GRU 모델 (메모리 효율이 더 좋음)
# # ============================================================
# model = Sequential([
#     GRU(64, return_sequences=True, input_shape=(TIMESTEPS, x_train.shape[2])),
#     Dropout(0.2),
    
#     GRU(32, return_sequences=False),
#     Dropout(0.2),
    
#     Dense(32, activation='relu'),
#     Dense(PREDICT_STEPS)
# ])

# model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# model.summary()


# # ============================================================
# # 6. 콜백 & 학습
# # ============================================================
# es = EarlyStopping(
#     monitor='val_loss',
#     patience=12,
#     restore_best_weights=True,
#     verbose=1
# )

# rlr = ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.5,
#     patience=6,
#     verbose=1
# )

# history = model.fit(
#     x_train, y_train,
#     validation_data=(x_val, y_val),
#     epochs=100,
#     batch_size=64,          # 메모리 여유 있으면 128까지 가능
#     callbacks=[es, rlr],
#     verbose=1
# )


# # ============================================================
# # 7. 평가
# # ============================================================
# y_pred_scaled = model.predict(x_test)

# # 원래 스케일로 복원
# y_test_orig = scaler_y.inverse_transform(y_test)
# y_pred_orig = scaler_y.inverse_transform(y_pred_scaled)

# rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
# mae  = mean_absolute_error(y_test_orig, y_pred_orig)

# print(f"\n===== Test 성능 =====")
# print(f"RMSE : {rmse:.4f}")
# print(f"MAE  : {mae:.4f}")


# # ============================================================
# # 8. 최종 추론 (가장 마지막 시퀀스)
# # ============================================================
# last_sequence = x_seq[-1:]                    # (1, 6, 13)
# y_final_pred_scaled = model.predict(last_sequence)
# y_final_pred = scaler_y.inverse_transform(y_final_pred_scaled)

# print("\n===== 최종 1시간 예측 결과 (wd deg) =====")
# print(y_final_pred[0])


# # ============================================================
# # (선택) 학습 곡선
# # ============================================================
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='train_loss')
# plt.plot(history.history['val_loss'], label='val_loss')
# plt.title('Loss')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['mae'], label='train_mae')
# plt.plot(history.history['val_mae'], label='val_mae')
# plt.title('MAE')
# plt.legend()
# plt.tight_layout()
# plt.show()
####################################################################################################
# Epoch 16: early stopping
# Restoring model weights from the end of the best epoch: 4.
# 1972/1972 ━━━━━━━━━━━━━━━━━━━━ 2s 968us/step  

# ===== Test 성능 =====
# RMSE : 76.0422
# MAE  : 58.1739
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step

# ===== 최종 1시간 예측 결과 (wd deg) =====
# [174.53755 173.70688 174.76015 175.84895 171.24004 173.21835]

######################################################################################################
# ============================================================
#           Conv1D Test Performance
# ============================================================
# MSE       : 6207.2910
# MAE       : 60.8133
# RMSE      : 78.7864
# R²        : 0.0378
# ±5° 정확도  : 6.47%
# ±10° 정확도 : 12.62%
# ============================================================

# ===== 마지막 입력 시퀀스 =====
# shape: (1, 6, 13)


# ============================================================
#        최종 1시간 풍향 예측
# ============================================================
# 1번째 10분 후 : 172.51°
# 2번째 10분 후 : 173.39°
# 3번째 10분 후 : 173.57°
# 4번째 10분 후 : 173.21°
# 5번째 10분 후 : 172.43°
# 6번째 10분 후 : 173.29°
# ============================================================

































































































