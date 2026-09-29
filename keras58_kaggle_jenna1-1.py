import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# 1. 데이터 로드
# ============================================================
path = './_data/kaggle_jena/'

df = pd.read_csv(path + "jena_climate_2009_2016.csv", index_col=0)
print(df.shape)
print(df.columns)

# y = wd (deg)
y = df['wd (deg)'].values.reshape(-1, 1)

# x = wd를 제외한 나머지 특징
x = df.drop(['wd (deg)'], axis=1).values

print("x shape:", x.shape)
print("y shape:", y.shape)


# ============================================================
# 2. 스케일링
# ============================================================
scaler_x = StandardScaler()
# scaler_x = RobustScaler()   # 원하시면 RobustScaler로 변경 가능
scaler_y = StandardScaler()

x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y)


# ============================================================
# 3. 시퀀스 생성 함수 (1시간 → 다음 1시간 예측)
# ============================================================
def split_xy_sequences(x, y, timesteps=6, predict_steps=6):
    x_seq, y_seq = [], []
    
    for i in range(len(x) - timesteps - predict_steps + 1):
        x_seq.append(x[i : i + timesteps])
        y_seq.append(y[i + timesteps : i + timesteps + predict_steps].flatten())
    
    return np.array(x_seq), np.array(y_seq)


TIMESTEPS = 6
PREDICT_STEPS = 6

x_seq, y_seq = split_xy_sequences(x_scaled, y_scaled, 
                                  timesteps=TIMESTEPS, 
                                  predict_steps=PREDICT_STEPS)

print("===== 시퀀스 생성 결과 =====")
print("x_seq shape:", x_seq.shape)
print("y_seq shape:", y_seq.shape)


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
# 5. GRU 모델 구성
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
# 6. 콜백 설정 (EarlyStopping + ModelCheckpoint)
# ============================================================
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# ---------- 가중치 저장 파일명 만들기 ----------
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
print("현재 시간:", date)

save_path = './_save/jena_gru/'
# 폴더가 없으면 생성
import os
os.makedirs(save_path, exist_ok=True)

filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([save_path, "jena_gru_", date, "_", filename])

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=filepath,
    verbose=1
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    verbose=1
)


# ============================================================
# 7. 학습
# ============================================================
start_time = time.time()

hist = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[es, mcp, rlr],
    verbose=1
)

end_time = time.time()
print(f"\n총 학습 시간: {end_time - start_time:.2f} 초")


# ============================================================
# 8. 평가
# ============================================================
loss = model.evaluate(x_test, y_test, verbose=0)
print("\n===== Test Loss =====")
print("loss (mse):", loss[0])
print("mae      :", loss[1])

y_pred_scaled = model.predict(x_test)

# 원래 스케일로 복원
y_test_orig = scaler_y.inverse_transform(y_test)
y_pred_orig = scaler_y.inverse_transform(y_pred_scaled)

rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
mae  = mean_absolute_error(y_test_orig, y_pred_orig)
r2   = r2_score(y_test_orig, y_pred_orig)

print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R2   : {r2:.4f}")


# ============================================================
# 9. 최종 추론 + Submission 파일 생성
# ============================================================
# 가장 마지막 시퀀스로 다음 1시간 예측
last_sequence = x_seq[-1:]                      # (1, 6, 13)
y_final_pred_scaled = model.predict(last_sequence)
y_final_pred = scaler_y.inverse_transform(y_final_pred_scaled)

print("\n===== 최종 1시간 예측 결과 (wd deg) =====")
print(y_final_pred[0])

# ---------- Submission 파일 저장 ----------
# 예측 결과를 DataFrame으로 만들어 저장
submission = pd.DataFrame({
    'timestep': [f't+{i+1}' for i in range(PREDICT_STEPS)],
    'wd_pred': y_final_pred[0]
})

submission_path = f'./_save/jena_gru/jena_submission_{date}.csv'
submission.to_csv(submission_path, index=False)
print(f"\nSubmission 파일 저장 완료 → {submission_path}")
print(submission)


# ============================================================
# 10. 학습 곡선 시각화
# ============================================================
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'], label='train_loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist.history['mae'], label='train_mae')
plt.plot(hist.history['val_mae'], label='val_mae')
plt.title('MAE')
plt.legend()
plt.tight_layout()
plt.show()

#############################################################################################
# ===== 최종 1시간 예측 결과 (wd deg) =====
# [172.87796 173.73892 173.34859 173.25423 173.02034 172.09216]

# Submission 파일 저장 완료 → ./_save/jena_gru/jena_submission_0929_1641.csv
#   timestep     wd_pred
# 0      t+1  172.877960
# 1      t+2  173.738922
# 2      t+3  173.348587
# 3      t+4  173.254227
# 4      t+5  173.020340
# 5      t+6  172.092163
############################################################################################
# Epoch 20: early stopping
# Restoring model weights from the end of the best epoch: 5.

# 총 학습 시간: 693.63 초

# ===== Test Loss =====
# loss (mse): 0.7750915884971619
# mae      : 0.6641127467155457
# 1972/1972 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step    
# RMSE : 76.3139
# MAE  : 57.5664
# R2   : 0.0972
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step

# ===== 최종 1시간 예측 결과 (wd deg) =====
# [175.6157  174.67227 174.39163 174.56717 173.08366 172.30032]

# Submission 파일 저장 완료 → ./_save/jena_gru/jena_submission_0929_1712.csv
#   timestep     wd_pred
# 0      t+1  175.615707
# 1      t+2  174.672272
# 2      t+3  174.391632
# 3      t+4  174.567169
# 4      t+5  173.083664
# 5      t+6  172.300323
