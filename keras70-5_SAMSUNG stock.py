import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
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

# Unnamed 컬럼 제거
df1 = df1.loc[:, ~df1.columns.astype(str).str.contains("Unnamed")]
df2 = df2.loc[:, ~df2.columns.astype(str).str.contains("Unnamed")]
df3 = df3.loc[:, ~df3.columns.astype(str).str.contains("Unnamed")]

# 최근 1120개만 사용
TARGET_ROWS = 1120
df1 = df1.tail(TARGET_ROWS)
df2 = df2.tail(TARGET_ROWS)
df3 = df3.tail(TARGET_ROWS)

print("\n최근 1120개 적용 후")
print("아모레:", df1.shape)
print("하이닉스:", df2.shape)
print("삼성전자:", df3.shape)

# prefix 붙이기
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

# 전부 NaN인 컬럼 제거 (대비 컬럼이 문제)
df = df.dropna(axis=1, how="all")

print("전부 NaN 컬럼 제거 후 shape:", df.shape)
print("남은 컬럼:", df.columns.tolist())

feature_cols = [col for col in df.columns if col != "Target"]

# feature만 ffill
df[feature_cols] = df[feature_cols].ffill()

print("ffill 후 전체 NaN 개수:", df.isna().sum().sum())

# Target이 NaN인 마지막 행만 제거
data = df.dropna(subset=["Target"]).copy()

print("dropna(Target) 후 shape:", data.shape)

# 혹시 남은 NaN 행 제거
data = data.dropna(subset=feature_cols)

print("최종 data shape:", data.shape)

X = data[feature_cols].values.astype(np.float32)
y = data["Target"].values.astype(np.float32)

print("\nX:", X.shape)
print("y:", y.shape)
print("Feature 수:", len(feature_cols))

if len(X) == 0:
    print("\n!!! 데이터가 0개입니다. 컬럼별 NaN 개수를 확인합니다 !!!")
    print(df.isna().sum().sort_values(ascending=False).head(20))
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
# 9. 모델 구성
# ============================================================
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(TIMESTEPS, X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(16, activation="relu"))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# ============================================================
# 10. 컴파일, 훈련
# ============================================================
es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    shuffle=False,
    callbacks=[es],
    verbose=1
)

# ============================================================
# 11. 평가, 예측
# ============================================================
y_pred_scaled = model.predict(X_test, verbose=0).reshape(-1, 1)
y_pred = y_scaler.inverse_transform(y_pred_scaled).ravel()
y_test_actual = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

mae  = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
r2   = r2_score(y_test_actual, y_pred)

print("\n===== TEST PERFORMANCE =====")
print(f"MAE : {mae:,.2f} 원")
print(f"RMSE: {rmse:,.2f} 원")
print(f"R²  : {r2:.4f}")

# ============================================================
# 12. 시각화
# ============================================================
plt.figure(figsize=(10, 4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("LSTM Training History")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(y_test_actual, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.title("Samsung Next-Day Open Prediction")
plt.xlabel("Test Sample")
plt.ylabel("Price (KRW)")
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# 13. 다음 거래일 시가 예측
# ============================================================
last_sequence = X[-TIMESTEPS:]
last_sequence_scaled = x_scaler.transform(last_sequence).astype(np.float32)
last_sequence_scaled = last_sequence_scaled.reshape(1, TIMESTEPS, X.shape[1])

next_open_scaled = model.predict(last_sequence_scaled, verbose=0)
next_open_pred = y_scaler.inverse_transform(next_open_scaled.reshape(-1, 1))[0, 0]

last_date = data.index[-1]

print("\n===== NEXT TRADING DAY PREDICTION =====")
print("마지막 관측일:", last_date.strftime("%Y-%m-%d"))
print(f"예측 삼성전자 다음 거래일 시가: {next_open_pred:,.0f} 원")

# ============================================================
# 14. 모델 및 결과 저장
# ============================================================
save_path = path
model.save(save_path / "samsung_lstm_model.keras")

pd.DataFrame({
    "Target_Date": data.index[val_end:],
    "Actual_Next_Open": y_test_actual,
    "Predicted_Next_Open": y_pred
}).to_csv(save_path / "삼성전자_LSTM_test_results.csv", index=False, encoding="utf-8-sig")

pd.DataFrame({
    "Last_Observed_Date": [last_date.strftime("%Y-%m-%d")],
    "Predicted_Open_KRW": [next_open_pred],
    "TIMESTEPS": [TIMESTEPS],
    "Features": [X.shape[1]],
    "Test_MAE_KRW": [mae],
    "Test_RMSE_KRW": [rmse],
    "Test_R2": [r2]
}).to_csv(save_path / "삼성전자_다음거래일_시가예측.csv", index=False, encoding="utf-8-sig")

print("\n저장 완료")
print(save_path / "samsung_lstm_model.keras")
print(save_path / "삼성전자_LSTM_test_results.csv")
print(save_path / "삼성전자_다음거래일_시가예측.csv")