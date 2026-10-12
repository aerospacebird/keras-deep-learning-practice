
import os
import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import load_model

# ============================================================
# 0. 경로 설정
# ============================================================
data_path = Path("./_data/samsung/")
save_path = Path("./_data/_save/keras70/")

# ============================================================
# 1. 저장된 모델 / 스케일러 / 설정 불러오기
# ============================================================
print("===== 모델 및 설정 로드 =====")

# 모델 로드
lstm_model = load_model(save_path / "samsung_lstm_model.keras")
conv_model = load_model(save_path / "samsung_conv1d_model.keras")

# 스케일러 로드
x_scaler = joblib.load(save_path / "scaler_X.pkl")
y_scaler = joblib.load(save_path / "scaler_y.pkl")

# 설정 로드
with open(save_path / "model_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

TIMESTEPS = config["TIMESTEPS"]
feature_cols = config["feature_cols"]
best_w = config["lstm_weight"]
best_conv_w = config["conv1d_weight"]

print(f"TIMESTEPS     : {TIMESTEPS}")
print(f"Feature 수    : {len(feature_cols)}")
print(f"LSTM 가중치   : {best_w:.2f}")
print(f"Conv1D 가중치 : {best_conv_w:.2f}")
print(f"마지막 학습일 : {config['last_observation_date']}")

# ============================================================
# 2. 최신 데이터 로드 (예측용)
# ============================================================
def read_korean_csv(file_path, index_col=0):
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, index_col=index_col, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"CSV 인코딩을 판별할 수 없습니다: {file_path}")

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
    return df.sort_index()

print("\n===== 최신 데이터 로드 =====")
df1 = read_korean_csv(data_path / "아모레261012.csv")
df2 = read_korean_csv(data_path / "하이닉스261012.csv")
df3 = read_korean_csv(data_path / "삼성전자261012.csv")

df1 = normalize_date_index(df1, "아모레")
df2 = normalize_date_index(df2, "하이닉스")
df3 = normalize_date_index(df3, "삼성전자")

# 숫자형 변환
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

# Unnamed 제거 + prefix
df1 = df1.loc[:, ~df1.columns.astype(str).str.contains("Unnamed")]
df2 = df2.loc[:, ~df2.columns.astype(str).str.contains("Unnamed")]
df3 = df3.loc[:, ~df3.columns.astype(str).str.contains("Unnamed")]

df1 = df1.add_prefix("아모레_")
df2 = df2.add_prefix("하이닉스_")
df3 = df3.add_prefix("삼성전자_")

# 결합
df = pd.concat([df1, df2, df3], axis=1, join="inner").sort_index()
df = df.dropna(axis=1, how="all")          # 전부 NaN인 컬럼 제거
df = df.ffill()                            # 결측치 채우기

print("통합 데이터:", df.shape)
print("기간:", df.index.min(), "~", df.index.max())

# ============================================================
# 3. 예측용 입력 데이터 생성
#    - 학습 때 사용한 feature_cols와 동일한 컬럼만 사용
#    - 가장 최근 TIMESTEPS(30일) 데이터를 입력으로 사용
# ============================================================
# 학습 때 사용한 컬럼만 선택 (없는 컬럼이 있으면 에러)
missing_cols = [col for col in feature_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"필요한 컬럼이 없습니다: {missing_cols}")

latest_data = df[feature_cols].tail(TIMESTEPS)

if len(latest_data) != TIMESTEPS:
    raise ValueError(f"최신 데이터가 {TIMESTEPS}개보다 적습니다. 현재: {len(latest_data)}")

print(f"\n예측 입력 데이터 shape: {latest_data.shape}")
print(f"입력 기간: {latest_data.index[0]} ~ {latest_data.index[-1]}")

# 스케일링
latest_scaled = x_scaler.transform(latest_data.values).astype(np.float32)
latest_scaled = latest_scaled.reshape(1, TIMESTEPS, len(feature_cols))

# ============================================================
# 4. 예측
# ============================================================
print("\n===== 예측 중 =====")

next_lstm_scaled = lstm_model.predict(latest_scaled, verbose=0)
next_conv_scaled = conv_model.predict(latest_scaled, verbose=0)

next_lstm = y_scaler.inverse_transform(next_lstm_scaled)[0, 0]
next_conv = y_scaler.inverse_transform(next_conv_scaled)[0, 0]
next_ensemble = best_w * next_lstm + best_conv_w * next_conv

last_date = df.index[-1]

# ============================================================
# 5. 결과 출력
# ============================================================
print("\n===== NEXT TRADING DAY PREDICTION =====")
print(f"마지막 관측일          : {last_date.strftime('%Y-%m-%d')}")
print(f"LSTM 예측 시가         : {next_lstm:,.0f}원")
print(f"Conv1D 예측 시가       : {next_conv:,.0f}원")
print(f"Ensemble 예측 시가     : {next_ensemble:,.0f}원")
print(f"(LSTM 가중치 {best_w:.2f} / Conv1D 가중치 {best_conv_w:.2f})")

# ============================================================
# 6. 예측 결과 저장 (선택)
# ============================================================
result_df = pd.DataFrame([{
    "Last_Observation_Date": last_date.strftime("%Y-%m-%d"),
    "LSTM_Prediction": next_lstm,
    "Conv1D_Prediction": next_conv,
    "Ensemble_Prediction": next_ensemble,
    "LSTM_Weight": best_w,
    "Conv1D_Weight": best_conv_w
}])

result_df.to_csv(save_path / "predict_only_result.csv", index=False, encoding="utf-8-sig")
print(f"\n예측 결과 저장 완료: {save_path / 'predict_only_result.csv'}")