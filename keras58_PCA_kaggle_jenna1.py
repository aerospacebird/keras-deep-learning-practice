import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
# 3. PCA 분석 추가
# ============================================================
# 주성분 개수 설정 (원하는 대로 변경 가능)
n_components = 8

pca = PCA(n_components=n_components)
x_pca = pca.fit_transform(x_scaled)

print("\n===== PCA 결과 =====")
print("원본 특징 수:", x_scaled.shape[1])
print("PCA 후 특징 수:", x_pca.shape[1])
print("설명된 분산 비율 (각 주성분):")
print(pca.explained_variance_ratio_)
print("누적 설명 분산 비율:")
print(np.cumsum(pca.explained_variance_ratio_))
print(f"총 설명된 분산: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")


# (선택) PCA 설명 분산 시각화
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(range(1, n_components+1), pca.explained_variance_ratio_, alpha=0.7)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Explained Variance Ratio')

plt.subplot(1, 2, 2)
plt.plot(range(1, n_components+1), np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 4. 시퀀스 생성 함수 (1시간 → 다음 1시간 예측)
# ============================================================
def split_xy_sequences(x, y, timesteps=6, predict_steps=6):
    x_seq, y_seq = [], []
    
    for i in range(len(x) - timesteps - predict_steps + 1):
        x_seq.append(x[i : i + timesteps])
        y_seq.append(y[i + timesteps : i + timesteps + predict_steps].flatten())
    
    return np.array(x_seq), np.array(y_seq)


# ===== 파라미터 =====
TIMESTEPS = 6          # 과거 1시간
PREDICT_STEPS = 6      # 다음 1시간 예측

# PCA가 적용된 x_pca 사용
x_seq, y_seq = split_xy_sequences(x_pca, y_scaled, 
                                  timesteps=TIMESTEPS, 
                                  predict_steps=PREDICT_STEPS)

print("\n===== 시퀀스 생성 결과 =====")
print("x_seq shape:", x_seq.shape)   # (samples, 6, n_components)
print("y_seq shape:", y_seq.shape)   # (samples, 6)


# ============================================================
# 5. 시간 순서 유지하며 데이터 분리
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
# 6. GRU 모델
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
# 7. 콜백 & 학습
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
    batch_size=64,
    callbacks=[es, rlr],
    verbose=1
)


# ============================================================
# 8. 평가
# ============================================================
y_pred_scaled = model.predict(x_test)

y_test_orig = scaler_y.inverse_transform(y_test)
y_pred_orig = scaler_y.inverse_transform(y_pred_scaled)

rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
mae  = mean_absolute_error(y_test_orig, y_pred_orig)

print(f"\n===== Test 성능 =====")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")


# ============================================================
# 9. 최종 추론 (가장 마지막 시퀀스)
# ============================================================
last_sequence = x_seq[-1:]
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

########################################################################################################
# ===== PCA 결과 =====
# 원본 특징 수: 13
# PCA 후 특징 수: 8
# 설명된 분산 비율 (각 주성분):
# [6.15139485e-01 1.50038826e-01 1.30904175e-01 8.08573179e-02
#  1.30177740e-02 5.94034797e-03 3.96199426e-03 1.30087613e-04]
# 누적 설명 분산 비율:
# [0.61513948 0.76517831 0.89608249 0.9769398  0.98995758 0.99589793
#  0.99985992 0.99999001]
# 총 설명된 분산: 100.00%
