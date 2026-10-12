

# ============================================================
# SAMSUNG STOCK PREDICTION V3.0
#
# LSTM + Conv1D Weighted Ensemble
#
# 1. Korean CSV encoding support
# 2. Three-stock date alignment and merge
# 3. Next-trading-day Samsung open target
# 4. Time-series split
# 5. Train-only RobustScaler
# 6. LSTM + Conv1D training
# 7. Validation-based ensemble weight selection
# 8. Test metrics and baseline comparison
# 9. Next-trading-day prediction
# 10. Model, weights, scalers and config persistence
# 11. Model reload and prediction verification
# ============================================================

import os
import json
import random
from pathlib import Path

import joblib
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

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# 0. CONFIGURATION
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = Path("./_data/samsung/")
DATA_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "아모레": DATA_DIR / "아모레261012.csv",
    "하이닉스": DATA_DIR / "하이닉스261012.csv",
    "삼성전자": DATA_DIR / "삼성전자261012.csv"
}

TARGET_ROWS = 1120
TIMESTEPS = 30

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

TARGET_COL = "삼성전자_내일시가"

# ============================================================
# 1. CSV LOADER: Korean encoding support
# ============================================================

def read_korean_csv(file_path, index_col=0):
    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp949",
        "euc-kr"
    ]

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

        except UnicodeDecodeError:
            continue

    raise ValueError(
        f"CSV encoding error: {file_path}"
    )


# ============================================================
# 2. DATE INDEX NORMALIZATION
# ============================================================

def normalize_date_index(df, file_name):
    df = df.copy()

    original_index = pd.Series(
        df.index.astype(str).str.strip()
    )

    parsed = pd.to_datetime(
        original_index,
        errors="coerce"
    )

    # Fallback for YYYYMMDD strings
    invalid = parsed.isna()

    if invalid.any():
        parsed.loc[invalid] = pd.to_datetime(
            original_index.loc[invalid],
            format="%Y%m%d",
            errors="coerce"
        )

    if parsed.isna().any():
        raise ValueError(
            f"{file_name}: invalid date index. "
            "Check the first CSV column."
        )

    df.index = pd.DatetimeIndex(parsed)
    df.index.name = "날짜"

    if df.index.has_duplicates:
        raise ValueError(
            f"{file_name}: duplicate trading dates."
        )

    return df.sort_index()


# ============================================================
# 3. LOAD AND NORMALIZE THREE STOCK DATASETS
# ============================================================

stock_data = {}

for name, file_path in FILES.items():

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path.resolve()}"
        )

    df = read_korean_csv(file_path)
    df = normalize_date_index(df, file_path.name)

    # Numeric conversion
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.strip(),
                errors="coerce"
            )

    if "시가" not in df.columns:
        raise ValueError(
            f"{name}: '시가' column not found."
        )

    # Select the latest 1120 observations
    df = df.tail(TARGET_ROWS)

    if len(df) != TARGET_ROWS:
        raise ValueError(
            f"{name}: expected {TARGET_ROWS} rows, "
            f"got {len(df)}."
        )

    df = df.add_prefix(f"{name}_")
    stock_data[name] = df

    print(f"{name} Shape: {df.shape}")


# ============================================================
# 4. MERGE BY COMMON TRADING DATE
# ============================================================

df_all = pd.concat(
    [
        stock_data["아모레"],
        stock_data["하이닉스"],
        stock_data["삼성전자"]
    ],
    axis=1,
    join="inner"
).sort_index()

if df_all.empty:
    raise ValueError(
        "No common trading dates among the three stocks."
    )

print("\nMerged Shape:", df_all.shape)
print("First date:", df_all.index.min())
print("Last date :", df_all.index.max())

# Drop completely empty columns
df_all = df_all.dropna(axis=1, how="all")

# Remove irrelevant unnamed columns
feature_cols = [
    col for col in df_all.columns
    if not col.endswith("_Unnamed: 6")
]

# Ensure Samsung open is available
if "삼성전자_시가" not in feature_cols:
    raise ValueError("삼성전자_시가 column not found.")

# Target is next common trading day's Samsung opening price
df_all[TARGET_COL] = df_all["삼성전자_시가"].shift(-1)

# Save the target date associated with each input row
df_all["타깃날짜"] = pd.Series(
    df_all.index,
    index=df_all.index
).shift(-1)

# ============================================================
# 5. BUILD NUMERIC X AND Y
# ============================================================

# Only actual features go into X
feature_cols = [
    col for col in feature_cols
    if col != TARGET_COL
]

# Forward fill only: no backward interpolation
df_all[feature_cols] = df_all[feature_cols].ffill()

# Remove rows without usable inputs or target
valid_mask = (
    df_all[feature_cols].notna().all(axis=1)
    & df_all[TARGET_COL].notna()
)

X_df = df_all.loc[valid_mask, feature_cols].copy()
y_series = df_all.loc[valid_mask, TARGET_COL].copy()
target_dates = df_all.loc[valid_mask, "타깃날짜"].copy()

X_raw = X_df.to_numpy(dtype=np.float32)
y_raw = y_series.to_numpy(dtype=np.float32)

n = len(X_raw)
N_FEATURES = X_raw.shape[1]

if n <= TIMESTEPS + 20:
    raise ValueError("Not enough valid time-series data.")

if not np.isfinite(X_raw).all():
    raise ValueError("X contains NaN or infinite values.")

if not np.isfinite(y_raw).all():
    raise ValueError("Y contains NaN or infinite values.")

print("\n===== DATASET =====")
print("X:", X_raw.shape)
print("Y:", y_raw.shape)
print("Features:", N_FEATURES)
print("TIMESTEPS:", TIMESTEPS)

# Save merged dataset
df_all.to_csv(
    DATA_DIR / "삼성전자_통합데이터.csv",
    encoding="utf-8-sig"
)

# ============================================================
# 6. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

if not (TIMESTEPS < train_end < val_end < n):
    raise ValueError("Invalid chronological split boundaries.")

print("\n===== TIME SPLIT =====")
print("Train rows:", train_end)
print("Validation rows:", val_end - train_end)
print("Test rows:", n - val_end)

# ============================================================
# 7. FIT SCALERS USING TRAINING DATA ONLY
# ============================================================

x_scaler = RobustScaler()
y_scaler = RobustScaler()

x_scaler.fit(X_raw[:train_end])

y_scaler.fit(
    y_raw[:train_end].reshape(-1, 1)
)

X_scaled = x_scaler.transform(
    X_raw
).astype(np.float32)

y_scaled = y_scaler.transform(
    y_raw.reshape(-1, 1)
).astype(np.float32).ravel()

# ============================================================
# 8. CREATE SEQUENCES
#
# X[t-TIMESTEPS+1 : t+1] -> Y[t]
# ============================================================

def make_sequences(X, y, start, end, timesteps):
    X_seq = []
    y_seq = []
    target_positions = []

    for t in range(start, end):
        if t < timesteps - 1:
            continue

        X_seq.append(
            X[t - timesteps + 1:t + 1]
        )
        y_seq.append(y[t])
        target_positions.append(t)

    return (
        np.asarray(X_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32),
        np.asarray(target_positions, dtype=np.int64)
    )


X_train, y_train, pos_train = make_sequences(
    X_scaled, y_scaled,
    0, train_end, TIMESTEPS
)

X_val, y_val, pos_val = make_sequences(
    X_scaled, y_scaled,
    train_end, val_end, TIMESTEPS
)

X_test, y_test, pos_test = make_sequences(
    X_scaled, y_scaled,
    val_end, n, TIMESTEPS
)

print("\n===== LSTM / Conv1D INPUT SHAPES =====")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val  :", X_val.shape)
print("y_val  :", y_val.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)

assert X_train.shape[1:] == (TIMESTEPS, N_FEATURES)
assert X_val.shape[1:] == (TIMESTEPS, N_FEATURES)
assert X_test.shape[1:] == (TIMESTEPS, N_FEATURES)

# ============================================================
# 9. LSTM MODEL
# ============================================================

lstm_model = Sequential([
    Input(shape=(TIMESTEPS, N_FEATURES)),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1)
], name="LSTM_Stock_Model")

lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

# ============================================================
# 10. Conv1D MODEL
# ============================================================

conv1d_model = Sequential([
    Input(shape=(TIMESTEPS, N_FEATURES)),
    Conv1D(
        filters=64,
        kernel_size=3,
        padding="causal",
        activation="relu"
    ),
    Dropout(0.2),
    Conv1D(
        filters=32,
        kernel_size=3,
        padding="causal",
        activation="relu"
    ),
    GlobalAveragePooling1D(),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1)
], name="Conv1D_Stock_Model")

conv1d_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

print("\n===== LSTM MODEL =====")
lstm_model.summary()

print("\n===== Conv1D MODEL =====")
conv1d_model.summary()

# ============================================================
# 11. TRAIN BOTH MODELS
# ============================================================

def create_early_stopping():
    return EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

print("\n===== TRAIN LSTM =====")

history_lstm = lstm_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    shuffle=False,
    callbacks=[create_early_stopping()],
    verbose=1
)

print("\n===== TRAIN Conv1D =====")

history_conv = conv1d_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    shuffle=False,
    callbacks=[create_early_stopping()],
    verbose=1
)

# ============================================================
# 12. PREDICTION HELPER: RESTORE PRICE UNITS
# ============================================================

def predict_price(model, X, scaler):
    pred_scaled = model.predict(
        X, verbose=0
    ).reshape(-1, 1)

    return scaler.inverse_transform(
        pred_scaled
    ).ravel()


y_val_actual = y_scaler.inverse_transform(
    y_val.reshape(-1, 1)
).ravel()

y_test_actual = y_scaler.inverse_transform(
    y_test.reshape(-1, 1)
).ravel()

val_lstm = predict_price(
    lstm_model, X_val, y_scaler
)

val_conv = predict_price(
    conv1d_model, X_val, y_scaler
)

test_lstm = predict_price(
    lstm_model, X_test, y_scaler
)

test_conv = predict_price(
    conv1d_model, X_test, y_scaler
)

# ============================================================
# 13. SELECT ENSEMBLE WEIGHT ON VALIDATION ONLY
#
# y_ensemble = w * y_LSTM + (1-w) * y_Conv1D
# ============================================================

weight_grid = np.linspace(0.0, 1.0, 101)

weight_results = []

for w in weight_grid:
    pred = w * val_lstm + (1.0 - w) * val_conv

    weight_results.append({
        "LSTM_weight": float(w),
        "Conv1D_weight": float(1.0 - w),
        "Validation_MAE": float(
            mean_absolute_error(y_val_actual, pred)
        )
    })

weight_df = pd.DataFrame(weight_results)

best_row = weight_df.loc[
    weight_df["Validation_MAE"].idxmin()
]

BEST_W = float(best_row["LSTM_weight"])
BEST_CONV_W = 1.0 - BEST_W

print("\n===== ENSEMBLE WEIGHTS =====")
print(f"LSTM weight: {BEST_W:.2f}")
print(f"Conv1D weight: {BEST_CONV_W:.2f}")
print(
    f"Validation MAE: "
    f"{best_row['Validation_MAE']:,.2f} KRW"
)

val_ensemble = (
    BEST_W * val_lstm
    + BEST_CONV_W * val_conv
)

test_ensemble = (
    BEST_W * test_lstm
    + BEST_CONV_W * test_conv
)

# ============================================================
# 14. TEST METRICS + LAST-VALUE BASELINE
# ============================================================

def evaluate_regression(y_true, y_pred, name):
    return {
        "Model": name,
        "MAE_KRW": mean_absolute_error(y_true, y_pred),
        "RMSE_KRW": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
        "R2": r2_score(y_true, y_pred)
    }


# Baseline: today's observed Samsung open as next-day prediction
# At target position t, input's last row is the previous row.
# The corresponding prior open is recovered from X_raw.
open_col_idx = feature_cols.index("삼성전자_시가")

baseline_test = X_raw[pos_test - 1, open_col_idx]

results = pd.DataFrame([
    evaluate_regression(y_test_actual, test_lstm, "LSTM"),
    evaluate_regression(y_test_actual, test_conv, "Conv1D"),
    evaluate_regression(y_test_actual, test_ensemble, "Ensemble"),
    evaluate_regression(y_test_actual, baseline_test, "Last Open Baseline")
])

print("\n===== TEST PERFORMANCE =====")
print(results.to_string(index=False))

# ============================================================
# 15. PLOT LEARNING CURVES AND TEST PREDICTIONS
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(history_lstm.history["loss"], label="LSTM Train")
axes[0].plot(history_lstm.history["val_loss"], label="LSTM Val")
axes[0].set_title("LSTM Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history_conv.history["loss"], label="Conv Train")
axes[1].plot(history_conv.history["val_loss"], label="Conv Val")
axes[1].set_title("Conv1D Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MSE")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(y_test_actual, label="Actual", linewidth=2)
plt.plot(test_lstm, label="LSTM", alpha=0.8)
plt.plot(test_conv, label="Conv1D", alpha=0.8)
plt.plot(test_ensemble, label="Ensemble", linewidth=2)
plt.plot(baseline_test, label="Last Open Baseline", alpha=0.8)
plt.title("Samsung Stock: LSTM + Conv1D Ensemble")
plt.xlabel("Test Sample")
plt.ylabel("Price (KRW)")
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# 16. SAVE TEST RESULTS
# ============================================================

test_results = pd.DataFrame({
    "Input_Date": X_df.index[pos_test],
    "Target_Date": target_dates.iloc[pos_test].to_numpy(),
    "Actual_Next_Open_KRW": y_test_actual,
    "LSTM_Prediction_KRW": test_lstm,
    "Conv1D_Prediction_KRW": test_conv,
    "Ensemble_Prediction_KRW": test_ensemble,
    "Baseline_Prediction_KRW": baseline_test
})

test_results.to_csv(
    DATA_DIR / "samsung_ensemble_test.csv",
    index=False,
    encoding="utf-8-sig"
)

results.to_csv(
    DATA_DIR / "samsung_ensemble_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)

weight_df.to_csv(
    DATA_DIR / "samsung_ensemble_weight_search.csv",
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 17. NEXT TRADING DAY PREDICTION
# ============================================================

# Use latest observed feature rows, including the final row
# whose next-day target is not yet available.
latest_X = df_all[feature_cols].ffill().tail(TIMESTEPS)

if len(latest_X) != TIMESTEPS:
    raise ValueError("Insufficient rows for the final prediction.")

if latest_X.isna().any().any():
    raise ValueError("Missing values in the final prediction window.")

latest_scaled = x_scaler.transform(
    latest_X.to_numpy(dtype=np.float32)
).astype(np.float32)

latest_scaled = latest_scaled.reshape(
    1, TIMESTEPS, N_FEATURES
)

next_lstm = float(
    predict_price(lstm_model, latest_scaled, y_scaler)[0]
)

next_conv = float(
    predict_price(conv1d_model, latest_scaled, y_scaler)[0]
)

next_ensemble = float(
    BEST_W * next_lstm + BEST_CONV_W * next_conv
)

last_observed_date = df_all.index[-1]

print("\n===== NEXT TRADING DAY PREDICTION =====")
print("Last observed date:", last_observed_date.strftime("%Y-%m-%d"))
print(f"LSTM prediction    : {next_lstm:,.0f} KRW")
print(f"Conv1D prediction  : {next_conv:,.0f} KRW")
print(f"Ensemble prediction: {next_ensemble:,.0f} KRW")

prediction_result = pd.DataFrame([{
    "Last_Observed_Date": last_observed_date.strftime("%Y-%m-%d"),
    "Target": "Next Trading Day Open",
    "LSTM_Prediction_KRW": next_lstm,
    "Conv1D_Prediction_KRW": next_conv,
    "Ensemble_Prediction_KRW": next_ensemble,
    "LSTM_weight": BEST_W,
    "Conv1D_weight": BEST_CONV_W,
    "TIMESTEPS": TIMESTEPS,
    "N_FEATURES": N_FEATURES
}])

prediction_result.to_csv(
    DATA_DIR / "samsung_next_open_ensemble.csv",
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 18. SAVE FULL MODELS AND WEIGHTS
# ============================================================

# Full models: architecture + trained weights + compile config
lstm_model.save(
    DATA_DIR / "samsung_lstm.keras"
)

conv1d_model.save(
    DATA_DIR / "samsung_conv1d.keras"
)

# Weights only: load into the same model architecture
lstm_model.save_weights(
    DATA_DIR / "samsung_lstm.weights.h5"
)

conv1d_model.save_weights(
    DATA_DIR / "samsung_conv1d.weights.h5"
)

# Save scalers
joblib.dump(
    x_scaler,
    DATA_DIR / "x_scaler.pkl"
)

joblib.dump(
    y_scaler,
    DATA_DIR / "y_scaler.pkl"
)

# Save feature order and ensemble configuration
config = {
    "seed": SEED,
    "timesteps": TIMESTEPS,
    "n_features": N_FEATURES,
    "feature_columns": feature_cols,
    "target_column": TARGET_COL,
    "lstm_weight": BEST_W,
    "conv1d_weight": BEST_CONV_W,
    "last_observed_date": last_observed_date.strftime("%Y-%m-%d"),
    "train_ratio": TRAIN_RATIO,
    "validation_ratio": VAL_RATIO,
    "test_ratio": TEST_RATIO
}

with open(
    DATA_DIR / "ensemble_config.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(config, f, ensure_ascii=False, indent=4)

print("\n===== SAVED ARTIFACTS =====")

for path in sorted(DATA_DIR.iterdir()):
    if path.is_file():
        print(
            f"{path.name}: "
            f"{path.stat().st_size / (1024**2):.3f} MB"
        )

# ============================================================
# 19. RELOAD SAVED MODELS AND VERIFY PREDICTION
# ============================================================

loaded_lstm = load_model(
    DATA_DIR / "samsung_lstm.keras"
)

loaded_conv = load_model(
    DATA_DIR / "samsung_conv1d.keras"
)

loaded_x_scaler = joblib.load(
    DATA_DIR / "x_scaler.pkl"
)

loaded_y_scaler = joblib.load(
    DATA_DIR / "y_scaler.pkl"
)

with open(
    DATA_DIR / "ensemble_config.json",
    "r",
    encoding="utf-8"
) as f:
    loaded_config = json.load(f)

# Confirm reloaded predictions match saved-model predictions
reload_lstm = float(
    predict_price(
        loaded_lstm,
        latest_scaled,
        loaded_y_scaler
    )[0]
)

reload_conv = float(
    predict_price(
        loaded_conv,
        latest_scaled,
        loaded_y_scaler
    )[0]
)

reload_ensemble = (
    loaded_config["lstm_weight"] * reload_lstm
    + loaded_config["conv1d_weight"] * reload_conv
)

print("\n===== RELOAD VERIFICATION =====")
print(f"Original ensemble : {next_ensemble:,.6f} KRW")
print(f"Reloaded ensemble : {reload_ensemble:,.6f} KRW")
print(
    "Absolute difference:",
    f"{abs(next_ensemble - reload_ensemble):.10f} KRW"
)

assert np.isclose(
    next_ensemble,
    reload_ensemble,
    rtol=1e-5,
    atol=1e-3
), "Reloaded model prediction mismatch."

print("\nSUCCESS: models, weights, scalers and config saved.")
print("SUCCESS: reload verification passed.")

########################################################## result ###################################################
# ===== ENSEMBLE WEIGHTS =====
# LSTM weight: 0.00
# Conv1D weight: 1.00
# Validation MAE: 12,161.34 KRW

# ===== TEST PERFORMANCE =====
#              Model       MAE_KRW      RMSE_KRW         R2
#               LSTM 168145.109375 176080.063244 -10.669743
#             Conv1D  51049.785156  58975.314158  -0.309126
#           Ensemble  51049.785156  58975.314158  -0.309126
# Last Open Baseline  14356.547852  18607.036948   0.869685

# ===== NEXT TRADING DAY PREDICTION =====
# Last observed date: 2026-10-12
# LSTM prediction    : 82,173 KRW
# Conv1D prediction  : 212,066 KRW
# Ensemble prediction: 212,066 KRW

# ===== SAVED ARTIFACTS =====
# ensemble_config.json: 0.002 MB
# samsung_conv1d.keras: 0.219 MB
# samsung_conv1d.weights.h5: 0.079 MB
# samsung_ensemble_metrics.csv: 0.000 MB
# samsung_ensemble_test.csv: 0.011 MB
# samsung_ensemble_weight_search.csv: 0.003 MB
# samsung_lstm.keras: 0.512 MB
# samsung_lstm.weights.h5: 0.176 MB
# samsung_lstm_model.keras: 0.512 MB
# samsung_next_open_ensemble.csv: 0.000 MB
# x_scaler.pkl: 0.001 MB
# y_scaler.pkl: 0.001 MB
# 삼성전자261012.csv: 0.176 MB
# 삼성전자_LSTM_test_results.csv: 0.005 MB
# 삼성전자_다음거래일_시가예측.csv: 0.000 MB
# 삼성전자_통합데이터.csv: 0.353 MB
# 아모레261012.csv: 0.297 MB
# 하이닉스261012.csv: 0.200 MB

# ===== RELOAD VERIFICATION =====
# Original ensemble : 212,065.906250 KRW
# Reloaded ensemble : 212,065.906250 KRW
# Absolute difference: 0.0000000000 KRW

# SUCCESS: models, weights, scalers and config saved.
# SUCCESS: reload verification passed.

##################################################################################################
# 메일제목 : 이동욱  추론가격

# 첨부1,2 predict 한 코드/ 원코드
# 첨부3  weight_df
# 제한시간: 15:50===== ENSEMBLE WEIGHTS =====
# LSTM weight: 0.00
# Conv1D weight: 1.00
# Validation MAE: 12,161.34 KRW

# ===== TEST PERFORMANCE =====
#              Model       MAE_KRW      RMSE_KRW         R2
#               LSTM 168145.109375 176080.063244 -10.669743
#             Conv1D  51049.785156  58975.314158  -0.309126
#           Ensemble  51049.785156  58975.314158  -0.309126
# Last Open Baseline  14356.547852  18607.036948   0.869685

# ===== NEXT TRADING DAY PREDICTION =====
# Last observed date: 2026-10-12
# LSTM prediction    : 82,173 KRW
# Conv1D prediction  : 212,066 KRW
# Ensemble prediction: 212,066 KRW

# ===== SAVED ARTIFACTS =====
# ensemble_config.json: 0.002 MB
# samsung_conv1d.keras: 0.219 MB
# samsung_conv1d.weights.h5: 0.079 MB
# samsung_ensemble_metrics.csv: 0.000 MB
# samsung_ensemble_test.csv: 0.011 MB
# samsung_ensemble_weight_search.csv: 0.003 MB
# samsung_lstm.keras: 0.512 MB
# samsung_lstm.weights.h5: 0.176 MB
# samsung_lstm_model.keras: 0.512 MB
# samsung_next_open_ensemble.csv: 0.000 MB
# x_scaler.pkl: 0.001 MB
# y_scaler.pkl: 0.001 MB
# 삼성전자261012.csv: 0.176 MB
# 삼성전자_LSTM_test_results.csv: 0.005 MB
# 삼성전자_다음거래일_시가예측.csv: 0.000 MB
# 삼성전자_통합데이터.csv: 0.353 MB
# 아모레261012.csv: 0.297 MB
# 하이닉스261012.csv: 0.200 MB

# ===== RELOAD VERIFICATION =====
# Original ensemble : 212,065.906250 KRW
# Reloaded ensemble : 212,065.906250 KRW
# Absolute difference: 0.0000000000 KRW

# SUCCESS: models, weights, scalers and config saved.
# SUCCESS: reload verification passed.