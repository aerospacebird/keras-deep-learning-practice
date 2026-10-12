import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import json

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Conv1D, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# 0. 재현성 설정
# ============================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# 1. 데이터
# ============================================================
path = Path("./_data/samsung/")
save_path = Path("./_data/_save/keras70/")
save_path.mkdir(parents=True, exist_ok=True)

def read_korean_csv(file_path, index_col=0):
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, index_col=index_col, encoding=encoding)
            print(f"[CSV] {file_path.name} | encoding={encoding} | shape={df.shape}")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"CSV 인코딩을 판별할 수 없습니다: {file_path}")

df1 = read_korean_csv(path / "아모레261012.csv")
df2 = read_korean_csv(path / "하이닉스261012.csv")
df3 = read_korean_csv(path / "삼성전자261012.csv")

print("아모레:", df1.shape)
print("하이닉스:", df2.shape)
print("삼성전자:", df3.shape)

# ============================================================
# 2. 날짜 인덱스 처리
# ============================================================
def normalize_date_index(df, name):
    df = df.copy()
    original_index = pd.Index(df.index).astype(str).str.strip()
    
    parsed = pd.to_datetime(original_index, errors="coerce")
    invalid = parsed.isna()
    
    if invalid.any():
        parsed_yyyymmdd = pd.to_datetime(original_index[invalid], format="%Y%m%d", errors="coerce")
        parsed.loc[invalid] = parsed_yyyymmdd
    
    if parsed.isna().any():
        raise ValueError(f"{name}: 일부 날짜가 유효하지 않습니다.")
    
    df.index = pd.DatetimeIndex(parsed)
    df.index.name = "날짜"
    return df.sort_index()

df1 = normalize_date_index(df1, "아모레")
df2 = normalize_date_index(df2, "하이닉스")
df3 = normalize_date_index(df3, "삼성전자")

# ============================================================
# 3. 숫자형 변환 + Unnamed 제거 + prefix 붙이기
# ============================================================
for df in [df1, df2, df3]:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.strip(),
                errors="coerce"
            )

df1 = df1.loc[:, ~df1.columns.astype(str).str.contains("Unnamed")]
df2 = df2.loc[:, ~df2.columns.astype(str).str.contains("Unnamed")]
df3 = df3.loc[:, ~df3.columns.astype(str).str.contains("Unnamed")]

TARGET_ROWS = 1120
df1 = df1.tail(TARGET_ROWS)
df2 = df2.tail(TARGET_ROWS)
df3 = df3.tail(TARGET_ROWS)

print("\n최근 1120개 적용 후")
print("아모레:", df1.shape)
print("하이닉스:", df2.shape)
print("삼성전자:", df3.shape)

df1 = df1.add_prefix("아모레_")
df2 = df2.add_prefix("하이닉스_")
df3 = df3.add_prefix("삼성전자_")

print("\nprefix 후 삼성전자 컬럼:")
print(df3.columns.tolist())
print("삼성전자_시가 존재:", "삼성전자_시가" in df3.columns)

# ============================================================
# 4. 데이터 결합
# ============================================================
df = pd.concat([df1, df2, df3], axis=1, join="inner").sort_index()

print("\n통합 데이터:", df.shape)
print("기간:", df.index.min(), "~", df.index.max())

if "삼성전자_시가" not in df.columns:
    raise ValueError("삼성전자_시가 컬럼이 없습니다.")

# ============================================================
# 5. 다음 거래일 시가 정답 생성
# ============================================================
df["Target"] = df["삼성전자_시가"].shift(-1)

print("\nTarget 생성 후")
print("df shape:", df.shape)
print("Target NaN 개수:", df["Target"].isna().sum())
print("전체 NaN 개수:", df.isna().sum().sum())

# 전부 NaN인 컬럼 제거 (대비 컬럼)
df = df.dropna(axis=1, how="all")
print("전부 NaN 컬럼 제거 후 shape:", df.shape)

feature_cols = [col for col in df.columns if col != "Target"]
df[feature_cols] = df[feature_cols].ffill()

print("ffill 후 전체 NaN 개수:", df.isna().sum().sum())

data = df.dropna(subset=["Target"]).copy()
print("dropna(Target) 후 shape:", data.shape)

data = data.dropna(subset=feature_cols)
print("최종 data shape:", data.shape)

X = data[feature_cols].values.astype(np.float32)
y = data["Target"].values.astype(np.float32)

print("\nX:", X.shape)
print("y:", y.shape)
print("Feature 수:", len(feature_cols))

if len(X) == 0:
    raise ValueError("학습 데이터가 없습니다.")

# ============================================================
# 6. 시간 순서 분할 (Train 70% / Val 15% / Test 15%)
# ============================================================
n = len(data)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

print("train_end:", train_end, "val_end:", val_end, "n:", n)

# ============================================================
# 7. Scaler (학습 구간에만 fit)
# ============================================================
x_scaler = RobustScaler()
y_scaler = RobustScaler()

x_scaler.fit(X[:train_end])
y_scaler.fit(y[:train_end].reshape(-1, 1))

X_scaled = x_scaler.transform(X).astype(np.float32)
y_scaled = y_scaler.transform(y.reshape(-1, 1)).ravel().astype(np.float32)

# ============================================================
# 8. 시계열 데이터 생성 (timesteps=30)
# ============================================================
TIMESTEPS = 30

def make_sequences(X, y, start, end, timesteps):
    X_seq, y_seq = [], []
    for t in range(start, end):
        if t < timesteps - 1:
            continue
        X_seq.append(X[t - timesteps + 1 : t + 1])
        y_seq.append(y[t])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)

X_train, y_train = make_sequences(X_scaled, y_scaled, 0, train_end, TIMESTEPS)
X_val,   y_val   = make_sequences(X_scaled, y_scaled, train_end, val_end, TIMESTEPS)
X_test,  y_test  = make_sequences(X_scaled, y_scaled, val_end, n, TIMESTEPS)

print("\n시퀀스 데이터")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val  :", X_val.shape, "y_val:", y_val.shape)
print("X_test :", X_test.shape, "y_test:", y_test.shape)

# ============================================================
# 9. LSTM 모델
# ============================================================
lstm_model = Sequential()
lstm_model.add(LSTM(64, return_sequences=True, input_shape=(TIMESTEPS, X_train.shape[2])))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(32))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(16, activation="relu"))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
print("\n===== LSTM MODEL =====")
lstm_model.summary()

# ============================================================
# 10. Conv1D 모델
# ============================================================
conv_model = Sequential()
conv_model.add(Conv1D(64, kernel_size=3, padding="causal", activation="relu",
                      input_shape=(TIMESTEPS, X_train.shape[2])))
conv_model.add(Dropout(0.2))
conv_model.add(Conv1D(32, kernel_size=3, padding="causal", activation="relu"))
conv_model.add(GlobalAveragePooling1D())
conv_model.add(Dense(32, activation="relu"))
conv_model.add(Dropout(0.2))
conv_model.add(Dense(1))

conv_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
print("\n===== Conv1D MODEL =====")
conv_model.summary()

# ============================================================
# 11. 컴파일, 훈련
# ============================================================
es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

print("\n===== LSTM TRAINING =====")
history_lstm = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    shuffle=False,
    callbacks=[es],
    verbose=1
)

print("\n===== Conv1D TRAINING =====")
history_conv = conv_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    shuffle=False,
    callbacks=[es],
    verbose=1
)

# ============================================================
# 12. 검증 데이터로 앙상블 가중치 탐색
# ============================================================
val_lstm = y_scaler.inverse_transform(lstm_model.predict(X_val, verbose=0)).ravel()
val_conv = y_scaler.inverse_transform(conv_model.predict(X_val, verbose=0)).ravel()
y_val_actual = y_scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()

weights = np.linspace(0.0, 1.0, 101)
ensemble_val_mae = []

for w in weights:
    ensemble_val = w * val_lstm + (1.0 - w) * val_conv
    ensemble_val_mae.append(mean_absolute_error(y_val_actual, ensemble_val))

best_idx = int(np.argmin(ensemble_val_mae))
best_w = float(weights[best_idx])
best_conv_w = 1.0 - best_w

print("\n===== ENSEMBLE WEIGHT =====")
print(f"LSTM weight  : {best_w:.2f}")
print(f"Conv1D weight: {best_conv_w:.2f}")
print(f"Validation Ensemble MAE: {ensemble_val_mae[best_idx]:,.2f}원")

# ============================================================
# 13. 테스트 데이터 예측 + 평가
# ============================================================
test_lstm = y_scaler.inverse_transform(lstm_model.predict(X_test, verbose=0)).ravel()
test_conv = y_scaler.inverse_transform(conv_model.predict(X_test, verbose=0)).ravel()
test_ensemble = best_w * test_lstm + best_conv_w * test_conv
y_test_actual = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

results = []
for name, pred in [("LSTM", test_lstm), ("Conv1D", test_conv), ("Ensemble", test_ensemble)]:
    mae  = mean_absolute_error(y_test_actual, pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, pred))
    r2   = r2_score(y_test_actual, pred)
    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

results_df = pd.DataFrame(results)
print("\n===== TEST PERFORMANCE =====")
print(results_df.round(2))

# ============================================================
# 14. 시각화
# ============================================================
plt.figure(figsize=(12, 5))
plt.plot(history_lstm.history["val_loss"], label="LSTM Val Loss")
plt.plot(history_conv.history["val_loss"], label="Conv1D Val Loss")
plt.title("Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(15, 6))
plt.plot(y_test_actual, label="Actual")
plt.plot(test_lstm, label="LSTM")
plt.plot(test_conv, label="Conv1D")
plt.plot(test_ensemble, label="Ensemble")
plt.title("Samsung Next-Day Open: Model Comparison")
plt.xlabel("Test Sample")
plt.ylabel("Price (KRW)")
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# 15. 다음 거래일 시가 예측
# ============================================================
last_sequence = X[-TIMESTEPS:]
last_sequence_scaled = x_scaler.transform(last_sequence).astype(np.float32)
last_sequence_scaled = last_sequence_scaled.reshape(1, TIMESTEPS, X.shape[1])

next_lstm = y_scaler.inverse_transform(lstm_model.predict(last_sequence_scaled, verbose=0))[0, 0]
next_conv = y_scaler.inverse_transform(conv_model.predict(last_sequence_scaled, verbose=0))[0, 0]
next_ensemble = best_w * next_lstm + best_conv_w * next_conv

last_date = data.index[-1]

print("\n===== NEXT TRADING DAY PREDICTION =====")
print("마지막 관측일:", last_date.strftime("%Y-%m-%d"))
print(f"LSTM 예측 시가    : {next_lstm:,.0f}원")
print(f"Conv1D 예측 시가  : {next_conv:,.0f}원")
print(f"Ensemble 예측 시가: {next_ensemble:,.0f}원")

# ============================================================
# 16. 모델 및 결과 저장 (./study/_save/keras70/)
# ============================================================
# 모델 저장
lstm_model.save(save_path / "samsung_lstm_model.keras")
conv_model.save(save_path / "samsung_conv1d_model.keras")

# 가중치만 별도 저장
lstm_model.save_weights(save_path / "samsung_lstm_weights.weights.h5")
conv_model.save_weights(save_path / "samsung_conv1d_weights.weights.h5")

# Scaler 저장
joblib.dump(x_scaler, save_path / "scaler_X.pkl")
joblib.dump(y_scaler, save_path / "scaler_y.pkl")

# 설정 저장
config = {
    "TIMESTEPS": TIMESTEPS,
    "feature_cols": feature_cols,
    "n_features": len(feature_cols),
    "target": "삼성전자 다음 거래일 시가",
    "lstm_weight": best_w,
    "conv1d_weight": best_conv_w,
    "last_observation_date": last_date.strftime("%Y-%m-%d")
}

with open(save_path / "model_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=4)

# 성능 / 예측 결과 저장
results_df.to_csv(save_path / "test_metrics.csv", index=False, encoding="utf-8-sig")

pd.DataFrame({
    "LSTM_Weight": weights,
    "Conv1D_Weight": 1.0 - weights,
    "Validation_MAE": ensemble_val_mae
}).to_csv(save_path / "ensemble_weight_search.csv", index=False, encoding="utf-8-sig")

pd.DataFrame(history_lstm.history).to_csv(save_path / "lstm_history.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(history_conv.history).to_csv(save_path / "conv1d_history.csv", index=False, encoding="utf-8-sig")

pd.DataFrame({
    "Actual_Open": y_test_actual,
    "LSTM_Prediction": test_lstm,
    "Conv1D_Prediction": test_conv,
    "Ensemble_Prediction": test_ensemble
}).to_csv(save_path / "test_predictions.csv", index=False, encoding="utf-8-sig")

pd.DataFrame([{
    "Last_Observation_Date": last_date.strftime("%Y-%m-%d"),
    "LSTM_Prediction": next_lstm,
    "Conv1D_Prediction": next_conv,
    "Ensemble_Prediction": next_ensemble,
    "LSTM_Weight": best_w,
    "Conv1D_Weight": best_conv_w
}]).to_csv(save_path / "next_prediction.csv", index=False, encoding="utf-8-sig")

print("\n모델 및 결과 저장 완료:", save_path)

# ============================================================
# 17. 저장한 모델 불러오기 및 검증
# ============================================================
loaded_lstm = load_model(save_path / "samsung_lstm_model.keras")
loaded_conv = load_model(save_path / "samsung_conv1d_model.keras")

print("\n===== MODEL RESTORE CHECK =====")
print("LSTM 모델 복원:", np.allclose(
    lstm_model.predict(X_test[:5], verbose=0),
    loaded_lstm.predict(X_test[:5], verbose=0),
    rtol=1e-5, atol=1e-6
))
print("Conv1D 모델 복원:", np.allclose(
    conv_model.predict(X_test[:5], verbose=0),
    loaded_conv.predict(X_test[:5], verbose=0),
    rtol=1e-5, atol=1e-6
))

print("\nCRAZY AI SAMSUNG STOCK PREDICTION 완료")
########################################################################################################################
# ===== ENSEMBLE WEIGHT =====
# LSTM weight  : 0.00
# Conv1D weight: 1.00
# Validation Ensemble MAE: 12,161.34원

# ===== TEST PERFORMANCE =====
#       Model        MAE       RMSE     R2
# 0      LSTM  166800.53  174622.49 -10.48
# 1    Conv1D   51049.79   58975.31  -0.31
# 2  Ensemble   51049.79   58975.31  -0.31

# ===== NEXT TRADING DAY PREDICTION =====
# 마지막 관측일: 2026-10-08
# LSTM 예측 시가    : 83,211원
# Conv1D 예측 시가  : 211,211원
# Ensemble 예측 시가: 211,211원

# 모델 및 결과 저장 완료: _data\_save\keras70

# ===== MODEL RESTORE CHECK =====
# LSTM 모델 복원: True
# Conv1D 모델 복원: True

# CRAZY AI SAMSUNG STOCK PREDICTION 완료