
# 삼성전자 주가 맞히기 (2026, 10,13일)

import os
import time
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from sklearn.metrics import r2_score, mean_squared_error

# 30-1 카피
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import Dense, Dropout, Conv2D,Flatten, GlobalAveragePooling2D #💛💛💛💛💛
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# ============================================================
# 1. DATA
# ============================================================

#path = './_data/samsung/'




# ============================================================
# CSV 한글 인코딩 자동 처리
# ============================================================
from pathlib import Path
import pandas as pd

def read_korean_csv(file_path, index_col=0):
    file_path = Path(file_path)

    # 한국어 CSV에서 자주 사용하는 인코딩 순서
    encodings = [
        "utf-8-sig",  # UTF-8 BOM 포함
        "utf-8",      # 표준 UTF-8
        "cp949",      # 한국어 Windows Excel에서 흔함
        "euc-kr"      # 한국어 EUC-KR
    ]

    for encoding in encodings:
        try:
            df = pd.read_csv(
                file_path,
                index_col=index_col,
                encoding=encoding
            )

            # 인코딩 및 읽기 결과 확인
            print(f"파일: {file_path.name}")
            print(f"사용 인코딩: {encoding}")
            print(f"데이터 크기: {df.shape}")
            print(f"컬럼: {df.columns.tolist()}")

            return df

        except UnicodeDecodeError:
            continue

    raise ValueError(
        f"지원하는 인코딩으로 읽지 못했습니다: {file_path}"
    )


# ============================================================
# 데이터 로드
# ============================================================
path = Path("./_data/samsung/")

df1 = read_korean_csv(path / "아모레261012.csv")
df2 = read_korean_csv(path / "하이닉스261012.csv")
df3 = read_korean_csv(path / "삼성전자261012.csv")


print(df1.head()) #(2060, 16)
#컬럼: ['시가', '고가', '저가', '종가', '대비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)', 
#       '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비']

print(df2.head()) #(1300, 16)
#컬럼: ['시가', '고가', '저가', '종가', '대비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)',
#       '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비']

print(df3.head()) #(1120, 16)
# 컬럼: ['시가', '고가', '저가', '종가', '대비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)',
#        '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비']




# ============================================================
# CRAZY AI SAMSUNG STOCK PREDICTION V1.0
#
# 목표:
# 1. 삼성전자, SK하이닉스, 아모레퍼시픽 CSV 로드
# 2. 한글 CSV 인코딩 자동 처리
# 3. 각 종목 최근 1120행 선택
# 4. 날짜 기준 병합
# 5. 삼성전자 다음 거래일 시가를 Y로 설정
# 6. TIMESTEPS 기반 LSTM 데이터 생성
# 7. 시간 순서 Train / Validation / Test 분할
# 8. LSTM 학습 및 평가
# 9. 마지막 관측일 다음 거래일 시가 예측
# 10. 결과 CSV 저장
# ============================================================

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------------------------------------
# 0. 재현성 및 설정
# ------------------------------------------------------------
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = Path("./_data/samsung/")
DATA_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "아모레": DATA_DIR / "아모레261012.csv",
    "하이닉스": DATA_DIR / "하이닉스261012.csv",
    "삼성전자": DATA_DIR / "삼성전자261012.csv",
}

TARGET_ROWS = 1120
TIMESTEPS = 30

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

TARGET_COL = "삼성전자_내일시가"

# ------------------------------------------------------------
# 1. 한글 CSV 인코딩 자동 처리
# ------------------------------------------------------------
def read_korean_csv(file_path, index_col=0):
    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp949",
        "euc-kr",
    ]

    errors = []

    for encoding in encodings:
        try:
            df = pd.read_csv(
                file_path,
                index_col=index_col,
                encoding=encoding
            )

            print(
                f"[CSV] {file_path.name} | "
                f"encoding={encoding} | shape={df.shape}"
            )

            return df

        except UnicodeDecodeError as exc:
            errors.append(str(exc))

    raise ValueError(
        f"CSV 인코딩을 판별할 수 없습니다: {file_path}"
    )

# ------------------------------------------------------------
# 2. 날짜 인덱스 정규화
#    날짜 기준 병합이므로 실제 날짜가 필요함
# ------------------------------------------------------------
def normalize_date_index(df, file_name):
    df = df.copy()

    original_index = pd.Index(df.index).astype(str).str.strip()

    # 일반 날짜 형식 우선
    parsed = pd.to_datetime(
        original_index,
        errors="coerce"
    )

    # YYYYMMDD 형태의 날짜를 보완 처리
    invalid = parsed.isna()

    if invalid.any():
        parsed_yyyymmdd = pd.to_datetime(
            original_index[invalid],
            format="%Y%m%d",
            errors="coerce"
        )
        parsed.loc[invalid] = parsed_yyyymmdd

    valid_ratio = parsed.notna().mean()

    if valid_ratio < 0.95:
        raise ValueError(
            f"{file_name}: 날짜 인덱스를 확인할 수 없습니다. "
            "CSV 첫 번째 열이 실제 거래일인지 확인하세요."
        )

    if parsed.isna().any():
        raise ValueError(
            f"{file_name}: 일부 날짜가 유효하지 않습니다."
        )

    df.index = pd.DatetimeIndex(parsed)
    df.index.name = "날짜"

    if df.index.has_duplicates:
        raise ValueError(
            f"{file_name}: 중복 거래일이 있습니다."
        )

    return df.sort_index()

# ------------------------------------------------------------
# 3. 데이터 로드 및 Shape 통일
# ------------------------------------------------------------
data = {}

for name, file_path in FILES.items():
    if not file_path.exists():
        raise FileNotFoundError(
            f"파일이 없습니다: {file_path.resolve()}"
        )

    temp = read_korean_csv(file_path)
    temp = normalize_date_index(temp, file_path.name)

    # 숫자형 변환: 쉼표 및 퍼센트 기호 처리
    for col in temp.columns:
        if temp[col].dtype == "object":
            temp[col] = pd.to_numeric(
                temp[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.strip(),
                errors="coerce"
            )

    if "시가" not in temp.columns:
        raise ValueError(
            f"{name} 데이터에 '시가' 컬럼이 없습니다."
        )

    # 최근 1120개 거래일 선택
    temp = temp.tail(TARGET_ROWS)

    if len(temp) != TARGET_ROWS:
        raise ValueError(
            f"{name}: {TARGET_ROWS}개 행이 필요하지만 "
            f"{len(temp)}개만 존재합니다."
        )

    # 컬럼 이름 중복 방지
    temp = temp.add_prefix(f"{name}_")

    data[name] = temp

    print(f"{name} Shape: {temp.shape}")

# ------------------------------------------------------------
# 4. 날짜 기준 병합
#    세 종목 공통 거래일만 사용
# ------------------------------------------------------------
df_all = pd.concat(
    [
        data["아모레"],
        data["하이닉스"],
        data["삼성전자"],
    ],
    axis=1,
    join="inner"
).sort_index()

if df_all.empty:
    raise ValueError(
        "공통 거래일이 없습니다. 날짜 인덱스를 확인하세요."
    )

if "삼성전자_시가" not in df_all.columns:
    raise ValueError("삼성전자 시가 컬럼이 없습니다.")

print("\n병합 데이터 Shape:", df_all.shape)
print("첫 거래일:", df_all.index.min())
print("마지막 거래일:", df_all.index.max())
#exit()
# ------------------------------------------------------------
# 5. Y 생성
#
# X[t] = 날짜 t에 관측 가능한 세 종목 데이터
# Y[t] = 삼성전자의 다음 관측 거래일 시가
#
# shift(-1)은 날짜 오름차순 정렬 후 적용
# ------------------------------------------------------------
df_all[TARGET_COL] = df_all["삼성전자_시가"].shift(-1)

# 입력 컬럼 목록: 타깃 제외
feature_cols = [
    col for col in df_all.columns
    if col != TARGET_COL
    and not col.endswith("_Unnamed: 6")
    and df_all[col].notna().any()
]

# 입력 데이터 결측 처리
# 과거 관측값으로만 채움
df_all[feature_cols] = df_all[feature_cols].ffill()  #병합 데이터 Shape: (1120, 48)
#exit()
# ------------------------------------------------------------
# 6. X / Y 분리
# ------------------------------------------------------------
X_df = df_all[feature_cols].copy()
y_series = df_all[TARGET_COL].copy()

# 학습 가능한 날짜는 X와 Y가 모두 존재하는 행
valid_mask = (
    X_df.notna().all(axis=1)
    & y_series.notna()
)

X_df = X_df.loc[valid_mask]
y_series = y_series.loc[valid_mask]

X_raw = X_df.to_numpy(dtype=np.float32)
y_raw = y_series.to_numpy(dtype=np.float32)

n = len(X_raw)
n_features = X_raw.shape[1]

if n <= TIMESTEPS + 10:
    raise ValueError(
        "시계열 학습 데이터가 부족합니다."
    )

if not np.isfinite(X_raw).all():
    raise ValueError("X에 NaN 또는 무한대가 남아 있습니다.")

if not np.isfinite(y_raw).all():
    raise ValueError("Y에 NaN 또는 무한대가 남아 있습니다.")

print("\nX Shape:", X_raw.shape)
print("Y Shape:", y_raw.shape)
print("Features:", n_features)

# X Shape: (1119, 42)
# Y Shape: (1119,)
# Features: 42
#exit()
# ------------------------------------------------------------
# 7. 시간 순서 기반 분할 경계
# ------------------------------------------------------------
train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

if not (TIMESTEPS < train_end < val_end < n):
    raise ValueError("Train / Validation / Test 분할을 확인하세요.")

# ------------------------------------------------------------
# 8. Train 데이터에만 Scaler fit
# ------------------------------------------------------------
x_scaler = RobustScaler()
y_scaler = RobustScaler()

# Train 타깃 구간까지의 입력 특성으로 fit
x_scaler.fit(X_raw[:train_end])

# Y의 학습 구간으로만 fit
y_scaler.fit(y_raw[:train_end].reshape(-1, 1))

# 모든 구간에는 Train에서 학습한 변환만 적용
X_scaled = x_scaler.transform(X_raw).astype(np.float32)

y_scaled = y_scaler.transform(
    y_raw.reshape(-1, 1)
).astype(np.float32).ravel()

# ------------------------------------------------------------
# 9. LSTM 시퀀스 생성
#
# X[t-TIMESTEPS+1 : t+1] -> Y[t]
# 과거 30개 행으로 다음 거래일 시가 예측
# ------------------------------------------------------------
def make_sequences(X, y, start, end, timesteps):
    X_seq = []
    y_seq = []

    for t in range(start, end):
        if t < timesteps - 1:
            continue

        X_seq.append(
            X[t - timesteps + 1:t + 1]
        )
        y_seq.append(y[t])

    return (
        np.asarray(X_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32)
    )

X_train, y_train = make_sequences(
    X_scaled, y_scaled,
    start=0, end=train_end,
    timesteps=TIMESTEPS
)

X_val, y_val = make_sequences(
    X_scaled, y_scaled,
    start=train_end, end=val_end,
    timesteps=TIMESTEPS
)

X_test, y_test = make_sequences(
    X_scaled, y_scaled,
    start=val_end, end=n,
    timesteps=TIMESTEPS
)

print("\n===== LSTM DATASET SHAPES =====")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val  :", X_val.shape)
print("y_val  :", y_val.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)
# ===== LSTM DATASET SHAPES =====
# X_train: (754, 30, 42)
# y_train: (754,)
# X_val  : (168, 30, 42)
# y_val  : (168,)
# X_test : (168, 30, 42)
# y_test : (168,)
#exit()
assert X_train.ndim == 3
assert X_val.ndim == 3
assert X_test.ndim == 3

assert X_train.shape[1:] == (TIMESTEPS, n_features)
assert X_val.shape[1:] == (TIMESTEPS, n_features)
assert X_test.shape[1:] == (TIMESTEPS, n_features)

# ------------------------------------------------------------
# 10. LSTM 모델
# ------------------------------------------------------------
model = Sequential([
    tf.keras.Input(
        shape=(TIMESTEPS, n_features)
    ),

    LSTM(
        64,
        return_sequences=True
    ),
    Dropout(0.2),

    LSTM(32),
    Dropout(0.2),

    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=["mae"]
)

model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# ------------------------------------------------------------
# 11. 학습
# ------------------------------------------------------------
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    shuffle=False,
    callbacks=[early_stop],
    verbose=1
)

# ------------------------------------------------------------
# 12. Test 평가: 원래 가격 단위로 복원
# ------------------------------------------------------------
y_pred_scaled = model.predict(
    X_test,
    verbose=0
).reshape(-1, 1)

y_pred = y_scaler.inverse_transform(
    y_pred_scaled
).ravel()

y_test_actual = y_scaler.inverse_transform(
    y_test.reshape(-1, 1)
).ravel()

mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(
    mean_squared_error(y_test_actual, y_pred)
)
r2 = r2_score(y_test_actual, y_pred)

print("\n===== TEST PERFORMANCE =====")
print(f"MAE : {mae:,.2f} 원")
print(f"RMSE: {rmse:,.2f} 원")
print(f"R²  : {r2:.4f}")

# 테스트 구간의 예측 결과 저장
test_result = pd.DataFrame({
    "Target_Date": X_df.index[val_end:],
    "Actual_Next_Open": y_test_actual,
    "Predicted_Next_Open": y_pred,
})

test_result.to_csv(
    DATA_DIR / "삼성전자_LSTM_test_results.csv",
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# 13. 학습 곡선
# ------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("LSTM Training History")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# 14. Test 실제값 vs 예측값
# ------------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(
    y_test_actual,
    label="Actual Next Open"
)
plt.plot(
    y_pred,
    label="Predicted Next Open"
)
plt.title("Samsung Electronics - LSTM Test Prediction")
plt.xlabel("Test Sample")
plt.ylabel("Price (KRW)")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# 15. 마지막 관측일 기준 다음 거래일 시가 예측
#
# 마지막 TIMESTEPS개의 관측 입력으로 예측
# ------------------------------------------------------------
last_sequence = X_raw[-TIMESTEPS:]

last_sequence_scaled = x_scaler.transform(
    last_sequence
).astype(np.float32)

last_sequence_scaled = last_sequence_scaled.reshape(
    1, TIMESTEPS, n_features
)

next_open_scaled = model.predict(
    last_sequence_scaled,
    verbose=0
)

next_open_pred = y_scaler.inverse_transform(
    next_open_scaled.reshape(-1, 1)
)[0, 0]

last_date = X_df.index[-1]

print("\n===== NEXT TRADING DAY PREDICTION =====")
print("마지막 관측일:", last_date.strftime("%Y-%m-%d"))
print("입력 TIMESTEPS:", TIMESTEPS)
print(f"예측 삼성전자 다음 거래일 시가: {next_open_pred:,.0f} 원")

# 모델 및 예측 결과 저장
model.save(
    DATA_DIR / "samsung_lstm_model.keras"
)

prediction_result = pd.DataFrame({
    "Last_Observed_Date": [last_date.strftime("%Y-%m-%d")],
    "Target": ["Next Trading Day Open"],
    "Predicted_Open_KRW": [next_open_pred],
    "TIMESTEPS": [TIMESTEPS],
    "Features": [n_features],
    "Test_MAE_KRW": [mae],
    "Test_RMSE_KRW": [rmse],
    "Test_R2": [r2],
})

prediction_result.to_csv(
    DATA_DIR / "삼성전자_다음거래일_시가예측.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n저장 완료:")
print(DATA_DIR / "samsung_lstm_model.keras")
print(DATA_DIR / "삼성전자_LSTM_test_results.csv")
print(DATA_DIR / "삼성전자_다음거래일_시가예측.csv")
