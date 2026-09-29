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

#1. 데이터
path = './_data/kaggle_jena/'
dataset = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)

print(dataset.shape)

y_cor = datasets[-144:]['wd (deg)']
print(y_cor.shape)
############################# 훈련 데이터 자르기  #################################
x_data= datasets[:-288].drop(['wd(deg)'], axis=1)
y_data= datasets[144:-144]['wd(deg)']

print(x_data.shape)
print(y_data.shape)

size_x= 144
size_y= 144

def split_x(datasets, size):

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


# ============================================================
# 5. GRU 모델 (메모리 효율이 더 좋음)
# ============================================================
model = Sequential([
    GRU(64, return_sequences=True, input_shape=(TIMESTEPS, x_train.shape[2])),
    Dropout(0.2),
    
    GRU(32, return_sequences=False),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    Dense(PREDICT_STEPS)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()


# ============================================================
# 6. 콜백 & 학습
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
    verbose=1
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=64,          # 메모리 여유 있으면 128까지 가능
    callbacks=[es, rlr],
    verbose=1
)


# ============================================================
# 7. 평가
# ============================================================
y_pred_scaled = model.predict(x_test)

# 원래 스케일로 복원
y_test_orig = scaler_y.inverse_transform(y_test)
y_pred_orig = scaler_y.inverse_transform(y_pred_scaled)

rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
mae  = mean_absolute_error(y_test_orig, y_pred_orig)

print(f"\n===== Test 성능 =====")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")


# ============================================================
# 8. 최종 추론 (가장 마지막 시퀀스)
# ============================================================
last_sequence = x_seq[-1:]                    # (1, 6, 13)
y_final_pred_scaled = model.predict(last_sequence)
y_final_pred = scaler_y.inverse_transform(y_final_pred_scaled)

print("\n===== 최종 1시간 예측 결과 (wd deg) =====")
print(y_final_pred[0])


# ============================================================
# (선택) 학습 곡선
# ============================================================
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='train_mae')
plt.plot(history.history['val_mae'], label='val_mae')
plt.title('MAE')
plt.legend()
plt.tight_layout()
plt.show()
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

































































































