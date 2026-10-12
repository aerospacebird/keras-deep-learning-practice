
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
#    - 시드를 고정해서 여러 번 실행해도 같은 결과가 나오도록 함
#    - PYTHONHASHSEED, random, numpy, tensorflow 모두 동일하게 맞춤
# ============================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# 1. 데이터 로드
#    - 삼성전자, SK하이닉스, 아모레퍼시픽 주가 CSV 파일을 읽음
#    - 한글 인코딩이 다를 수 있어서 utf-8-sig → utf-8 → cp949 → euc-kr 순서로 시도
# ============================================================
path = Path("./_data/samsung/")          # 원본 데이터 경로
save_path = Path("./_data/_save/keras70/")  # 모델/결과 저장 경로
save_path.mkdir(parents=True, exist_ok=True)  # 폴더가 없으면 생성

def read_korean_csv(file_path, index_col=0):
    """한글 CSV를 안전하게 읽는 함수"""
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
# 아모레: (2060, 16)
# 하이닉스: (1300, 16)
# 삼성전자: (1120, 16)
# ============================================================
# 2. 날짜 인덱스 처리
#    - CSV의 인덱스가 날짜 문자열이므로 datetime으로 변환
#    - 일반 날짜 형식과 YYYYMMDD 형식을 모두 처리
#    - 변환 실패한 날짜가 있으면 에러를 발생시킴
# ============================================================
def normalize_date_index(df, name):
    """날짜 인덱스를 DatetimeIndex로 정규화하는 함수"""
    df = df.copy()
    original_index = pd.Index(df.index).astype(str).str.strip()
    
    # 먼저 일반적인 날짜 형식으로 변환 시도
    parsed = pd.to_datetime(original_index, errors="coerce")
    invalid = parsed.isna()
    
    # 실패한 것들은 YYYYMMDD 형식으로 다시 시도
    if invalid.any():
        parsed_yyyymmdd = pd.to_datetime(original_index[invalid], format="%Y%m%d", errors="coerce")
        parsed.loc[invalid] = parsed_yyyymmdd
    
    # 여전히 실패한 날짜가 있으면 에러
    if parsed.isna().any():
        raise ValueError(f"{name}: 일부 날짜가 유효하지 않습니다.")
    
    df.index = pd.DatetimeIndex(parsed)
    df.index.name = "날짜"
    return df.sort_index()   # 날짜 오름차순 정렬

df1 = normalize_date_index(df1, "아모레")
df2 = normalize_date_index(df2, "하이닉스")
df3 = normalize_date_index(df3, "삼성전자")

# ============================================================
# 3. 숫자형 변환 + Unnamed 제거 + prefix 붙이기
#    - 문자열로 되어 있는 숫자(쉼표, % 포함)를 float로 변환
#    - Unnamed 컬럼(빈 컬럼) 제거
#    - 세 종목의 컬럼명이 겹치지 않도록 prefix 추가
#    - 최근 1120개 거래일만 사용 (가장 짧은 종목 기준)
# ============================================================
for df in [df1, df2, df3]:
    for col in df.columns:
        if df[col].dtype == "object":
            # 쉼표와 % 기호를 제거하고 숫자로 변환
            df[col] = pd.to_numeric(
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.strip(),
                errors="coerce"
            )

# Unnamed로 시작하는 쓸모없는 컬럼 제거
df1 = df1.loc[:, ~df1.columns.astype(str).str.contains("Unnamed")]
df2 = df2.loc[:, ~df2.columns.astype(str).str.contains("Unnamed")]
df3 = df3.loc[:, ~df3.columns.astype(str).str.contains("Unnamed")]

# 가장 짧은 데이터(삼성전자 1120개)에 맞춰 최근 데이터만 사용
TARGET_ROWS = 1120
df1 = df1.tail(TARGET_ROWS)
df2 = df2.tail(TARGET_ROWS)
df3 = df3.tail(TARGET_ROWS)

print("\n최근 1120개 적용 후")
print("아모레:", df1.shape)
print("하이닉스:", df2.shape)
print("삼성전자:", df3.shape)
# 최근 1120개 적용 후
# 아모레: (1120, 15)
# 하이닉스: (1120, 15)
# 삼성전자: (1120, 15)

# 컬럼명에 종목명을 prefix로 붙여서 구분
df1 = df1.add_prefix("아모레_")
df2 = df2.add_prefix("하이닉스_")
df3 = df3.add_prefix("삼성전자_")

print("\nprefix 후 삼성전자 컬럼:")
print(df3.columns.tolist())
print("삼성전자_시가 존재:", "삼성전자_시가" in df3.columns)

# ============================================================
# 4. 데이터 결합
#    - 세 종목을 날짜(index) 기준으로 가로로 합침
#    - join="inner" → 세 종목 모두에 존재하는 공통 거래일만 사용
# ============================================================
df = pd.concat([df1, df2, df3], axis=1, join="inner").sort_index()

print("\n통합 데이터:", df.shape) #통합 데이터: (1120, 45)
print("기간:", df.index.min(), "~", df.index.max()) #기간: 2022-03-14 00:00:00 ~ 2026-10-12 00:00:00

if "삼성전자_시가" not in df.columns:
    raise ValueError("삼성전자_시가 컬럼이 없습니다.")

# ============================================================
# 5. 다음 거래일 시가 정답(Target) 생성
#    - 목표: 오늘까지의 데이터로 "내일 삼성전자 시가"를 예측
#    - shift(-1) : 다음 행의 값을 현재 행으로 가져옴
#    - 전부 행은 Target이 NaN이 되므로 나중에 제거
#    - 전부 NaN인 컬럼(예: 대비)은 미리 제거
# ============================================================
df["Target"] = df["삼성전자_시가"].shift(-1)

print("\nTarget 생성 후")
print("df shape:", df.shape)
print("Target NaN 개수:", df["Target"].isna().sum())
print("전체 NaN 개수:", df.isna().sum().sum())

# 모든 값이 NaN인 컬럼 제거 (대비 컬럼이 여기에 해당)
df = df.dropna(axis=1, how="all")
print("전부 NaN 컬럼 제거 후 shape:", df.shape)

feature_cols = [col for col in df.columns if col != "Target"]

# 결측치를 앞의 값으로 채움 (미래 정보는 사용하지 않음)
df[feature_cols] = df[feature_cols].ffill()
print("ffill 후 전체 NaN 개수:", df.isna().sum().sum())

# Target이 NaN인 마지막 행 제거
data = df.dropna(subset=["Target"]).copy()
print("dropna(Target) 후 shape:", data.shape)

# feature에 남아있는 NaN 행도 제거
data = data.dropna(subset=feature_cols)
print("최종 data shape:", data.shape)

X = data[feature_cols].values.astype(np.float32)  # 입력 특성
y = data["Target"].values.astype(np.float32)      # 정답 (다음 날 시가)

print("\nX:", X.shape)
print("y:", y.shape)
print("Feature 수:", len(feature_cols))
# X: (1119, 42)
# y: (1119,)
# Feature 수: 42

if len(X) == 0:
    raise ValueError("학습 데이터가 없습니다.")

# ============================================================
# 6. 시간 순서 분할
#    - 시계열 데이터이므로 랜덤 셔플 금지
#    - 앞쪽 70% : Train
#    - 중간 15% : Validation
#    - 뒤쪽 15% : Test
# ============================================================
n = len(data)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

print("train_end:", train_end, "val_end:", val_end, "n:", n)

# ============================================================
# 7. Scaler
#    - RobustScaler : 이상치에 덜 민감 (중앙값과 IQR 사용)
#    - 반드시 Train 구간에만 fit → 미래 정보가 학습에 들어가지 않도록 함
#    - 그 후 전체 데이터에 동일하게 transform
# ============================================================
x_scaler = RobustScaler()
y_scaler = RobustScaler()

x_scaler.fit(X[:train_end])                     # 입력 스케일러는 Train만으로 학습
y_scaler.fit(y[:train_end].reshape(-1, 1))      # 정답 스케일러도 Train만으로 학습

X_scaled = x_scaler.transform(X).astype(np.float32)
y_scaled = y_scaler.transform(y.reshape(-1, 1)).ravel().astype(np.float32)

# ============================================================
# 8. 시계열 시퀀스 생성
#    - TIMESTEPS = 30 → 과거 30일 데이터로 다음 날 시가 예측
#    - X[t-29 : t+1] → y[t]
#    - make_sequences 함수로 Train / Val / Test 각각 생성
# ============================================================
TIMESTEPS = 30

def make_sequences(X, y, start, end, timesteps):
    """슬라이딩 윈도우로 시퀀스 데이터 생성"""
    X_seq, y_seq = [], []
    for t in range(start, end):
        if t < timesteps - 1:      # 앞쪽 timesteps-1개는 윈도우를 만들 수 없음
            continue
        X_seq.append(X[t - timesteps + 1 : t + 1])  # 과거 timesteps개
        y_seq.append(y[t])                          # 해당 시점의 정답
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)

X_train, y_train = make_sequences(X_scaled, y_scaled, 0, train_end, TIMESTEPS)
X_val,   y_val   = make_sequences(X_scaled, y_scaled, train_end, val_end, TIMESTEPS)
X_test,  y_test  = make_sequences(X_scaled, y_scaled, val_end, n, TIMESTEPS)

print("\n시퀀스 데이터")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val  :", X_val.shape, "y_val:", y_val.shape)
print("X_test :", X_test.shape, "y_test:", y_test.shape)
# X_train: (754, 30, 42) y_train: (754,)
# X_val  : (168, 30, 42) y_val: (168,)
# X_test : (168, 30, 42) y_test: (168,)
# ============================================================
# 9. LSTM 모델 구성
#    - 시계열의 장기 의존성을 학습하는 데 강점
#    - return_sequences=True → 다음 LSTM 층에 시퀀스 전체를 전달
# ============================================================
lstm_model = Sequential()
lstm_model.add(LSTM(64, return_sequences=True, input_shape=(TIMESTEPS, X_train.shape[2])))
lstm_model.add(Dropout(0.2))          # 과적합 방지
lstm_model.add(LSTM(32))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(16, activation="relu"))
lstm_model.add(Dense(1))              # 최종 출력 (다음 날 시가)

lstm_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
print("\n===== LSTM MODEL =====")
lstm_model.summary()
# ===== LSTM MODEL =====
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  lstm (LSTM)                 (None, 30, 64)            27392     
                                                                 
#  dropout (Dropout)           (None, 30, 64)            0         
                                                                 
#  lstm_1 (LSTM)               (None, 32)                12416     
                                                                 
#  dropout_1 (Dropout)         (None, 32)                0         
                                                                 
#  dense (Dense)               (None, 16)                528       
                                                                 
#  dense_1 (Dense)             (None, 1)                 17        
                                                                 
# =================================================================
# Total params: 40,353
# Trainable params: 40,353
# Non-trainable params: 0
# ============================================================
# 10. Conv1D 모델 구성
#    - 지역적인 패턴(짧은 구간의 움직임)을 잘 잡음
#    - causal padding → 미래 정보를 보지 않음 (시계열에 적합)
#    - GlobalAveragePooling1D → 시퀀스 차원을 평균내서 벡터로 만듦
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
# ===== Conv1D MODEL =====
# Model: "sequential_1"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv1d (Conv1D)             (None, 30, 64)            8128      
                                                                 
#  dropout_2 (Dropout)         (None, 30, 64)            0         
                                                                 
#  conv1d_1 (Conv1D)           (None, 30, 32)            6176      
                                                                 
#  global_average_pooling1d (G  (None, 32)               0         
#  lobalAveragePooling1D)                                          
                                                                 
#  dense_2 (Dense)             (None, 32)                1056      
                                                                 
#  dropout_3 (Dropout)         (None, 32)                0         
                                                                 
#  dense_3 (Dense)             (None, 1)                 33        
                                                                 
# =================================================================
# Total params: 15,393
# Trainable params: 15,393
# Non-trainable params: 0
# ============================================================
# 11. 모델 훈련
#    - EarlyStopping : val_loss가 10 epoch 동안 개선되지 않으면 중단
#    - restore_best_weights=True → 가장 좋았던 시점의 가중치로 복원
#    - shuffle=False → 시계열 순서 유지
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
# 12. 앙상블 가중치 탐색
#    - Validation 데이터에서 LSTM과 Conv1D의 최적 비율을 찾음
#    - 0.00 ~ 1.00까지 0.01 간격으로 시도
#    - MAE가 가장 낮은 가중치를 선택
# ============================================================
val_lstm = y_scaler.inverse_transform(lstm_model.predict(X_val, verbose=0)).ravel()
val_conv = y_scaler.inverse_transform(conv_model.predict(X_val, verbose=0)).ravel()
y_val_actual = y_scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()

weights = np.linspace(0.0, 1.0, 101)   # 0.00, 0.01, ..., 1.00
ensemble_val_mae = []

for w in weights:
    ensemble_val = w * val_lstm + (1.0 - w) * val_conv
    ensemble_val_mae.append(mean_absolute_error(y_val_actual, ensemble_val))

best_idx = int(np.argmin(ensemble_val_mae))
best_w = float(weights[best_idx])       # LSTM 가중치
best_conv_w = 1.0 - best_w              # Conv1D 가중치

print("\n===== ENSEMBLE WEIGHT =====")
print(f"LSTM weight  : {best_w:.2f}")
print(f"Conv1D weight: {best_conv_w:.2f}")
print(f"Validation Ensemble MAE: {ensemble_val_mae[best_idx]:,.2f}원")
# ===== ENSEMBLE WEIGHT =====
# LSTM weight  : 0.00
# Conv1D weight: 1.00
# Validation Ensemble MAE: 12,161.34원
# ============================================================
# 13. 테스트 데이터 예측 및 성능 평가
#    - 스케일링된 예측값을 원래 가격 단위로 되돌림
#    - LSTM / Conv1D / Ensemble 세 가지 성능을 비교
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
# ===== TEST PERFORMANCE =====
#       Model        MAE       RMSE     R2
# 0      LSTM  168580.78  176607.62 -10.74
# 1    Conv1D   51049.77   58975.30  -0.31
# 2  Ensemble   51049.77   58975.30  -0.31
# ============================================================
# 14. 시각화
#    - 학습 곡선(Validation Loss) 비교
#    - 테스트 구간 실제값 vs 모델별 예측값 비교
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
#    - 가장 최근 TIMESTEPS(30일) 데이터를 입력으로 사용
#    - LSTM / Conv1D / Ensemble 세 가지 예측값을 모두 출력
# ============================================================
last_sequence = X[-TIMESTEPS:]                                    # 원본 스케일
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
# ===== NEXT TRADING DAY PREDICTION =====
# 마지막 관측일: 2026-10-08
# LSTM 예측 시가    : 81,643원
# Conv1D 예측 시가  : 211,211원
# Ensemble 예측 시가: 211,211원
# ============================================================
# 16. 모델 및 결과 저장
#    저장 위치: ./_data/_save/keras70/
#    - 모델 전체 (.keras)
#    - 가중치만 (.weights.h5)
#    - Scaler (.pkl)
#    - 설정 (json)
#    - 성능/예측 결과 (csv)
# ============================================================
# 모델 전체 저장
lstm_model.save(save_path / "samsung_lstm_model.keras")
conv_model.save(save_path / "samsung_conv1d_model.keras")

# 가중치만 별도 저장
lstm_model.save_weights(save_path / "samsung_lstm_weights.weights.h5")
conv_model.save_weights(save_path / "samsung_conv1d_weights.weights.h5")

# Scaler 저장 (나중에 예측할 때 동일한 스케일 변환을 위해 필요)
joblib.dump(x_scaler, save_path / "scaler_X.pkl")
joblib.dump(y_scaler, save_path / "scaler_y.pkl")

# 모델 설정 저장
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

# 성능 지표 저장
results_df.to_csv(save_path / "test_metrics.csv", index=False, encoding="utf-8-sig")

# 앙상블 가중치 탐색 결과 저장
pd.DataFrame({
    "LSTM_Weight": weights,
    "Conv1D_Weight": 1.0 - weights,
    "Validation_MAE": ensemble_val_mae
}).to_csv(save_path / "ensemble_weight_search.csv", index=False, encoding="utf-8-sig")

# 학습 히스토리 저장
pd.DataFrame(history_lstm.history).to_csv(save_path / "lstm_history.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(history_conv.history).to_csv(save_path / "conv1d_history.csv", index=False, encoding="utf-8-sig")

# 테스트 예측 결과 저장
pd.DataFrame({
    "Actual_Open": y_test_actual,
    "LSTM_Prediction": test_lstm,
    "Conv1D_Prediction": test_conv,
    "Ensemble_Prediction": test_ensemble
}).to_csv(save_path / "test_predictions.csv", index=False, encoding="utf-8-sig")

# 다음 거래일 예측 결과 저장
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
#    - 저장 → 로드 → 예측값이 동일한지 확인
#    - 모델이 제대로 저장되었는지 검증하는 단계
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

########################################################## RESULTS ################################################
# ===== ENSEMBLE WEIGHT =====
# LSTM weight  : 0.00
# Conv1D weight: 1.00
# Validation Ensemble MAE: 12,161.34원

# ===== TEST PERFORMANCE =====
#       Model        MAE       RMSE     R2
# 0      LSTM  168580.78  176607.62 -10.74
# 1    Conv1D   51049.77   58975.30  -0.31
# 2  Ensemble   51049.77   58975.30  -0.31

# ===== NEXT TRADING DAY PREDICTION =====
# 마지막 관측일: 2026-10-08
# LSTM 예측 시가    : 81,643원
# Conv1D 예측 시가  : 211,211원
# Ensemble 예측 시가: 211,211원

# 모델 및 결과 저장 완료: _data\_save\keras70

# ===== MODEL RESTORE CHECK =====
# LSTM 모델 복원: True
# Conv1D 모델 복원: True

# CRAZY AI SAMSUNG STOCK PREDICTION 완료