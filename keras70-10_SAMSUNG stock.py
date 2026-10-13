# ============================================================
# CRAZY AI STOCK PREDICTION V3.2
# Samsung Electronics Next Trading Day Open Prediction
#
# Models:
#   1. LSTM
#   2. Conv1D
#   3. Transformer Encoder
#   4. Validation-selected Ensemble with Baseline Fallback
#   5. Baseline: Current Samsung Close
#
# Pipeline:
#   CSV -> Date Alignment -> Target Generation
#   -> Chronological Split -> Train-only Scaling
#   -> Sequence Generation -> Model Training
#   -> Return Target -> Validation Ensemble + Baseline -> Test Evaluation
#   -> Latest Forecast -> Save -> Reload Verification
#
# Important:
#   - Samsung trading calendar is the master calendar.
#   - Other stocks are aligned to Samsung dates.
#   - Forward fill uses only previously available values.
#   - Scalers are fitted using training data only.
#   - Test data is not used to select ensemble weights.
# ============================================================
import os
import json
import random
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LSTM,
    Conv1D,
    GlobalAveragePooling1D,
    Embedding,
    Add,
    LayerNormalization,
    MultiHeadAttention
)
from tensorflow.keras.callbacks import EarlyStopping
warnings.filterwarnings("ignore")
# ============================================================
# 0. CONFIGURATION
# ============================================================
SEED = 42
TIMESTEPS = 30
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
EPOCHS = 100
BATCH_SIZE = 32
DATA_DIR = "./_data/samsung/"
SAVE_DIR = "./_data/_save/keras70/"
FILES = {
    "아모레": "아모레261012.csv",
    "하이닉스": "하이닉스261012.csv",
    "삼성전자": "삼성전자261012.csv",
}
TARGET_COL = "삼성전자_시가"
BASELINE_COL = "삼성전자_종가"
os.makedirs(SAVE_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
print("=" * 65)
print("CRAZY AI STOCK PREDICTION V3.0")
print("=" * 65)
print("TensorFlow:", tf.__version__)
print("Data directory:", os.path.abspath(DATA_DIR))
print("Save directory:", os.path.abspath(SAVE_DIR))
# ============================================================
# 1. CSV LOADER
# ============================================================
def read_csv_korean(path):
    """Read CSV using common Korean encodings."""
    last_error = None
    for encoding in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(
                path,
                index_col=0,
                encoding=encoding
            )
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(
        f"Unable to decode CSV: {path}"
    ) from last_error
def normalize_date_index(df):
    """Convert index to datetime, remove invalid/duplicate dates."""
    df = df.copy()
    raw_index = pd.Index(
        df.index.astype(str).str.strip()
    )
    date_index = pd.to_datetime(
        raw_index,
        errors="coerce"
    )
    invalid = date_index.isna()
    if invalid.any():
        parsed = pd.to_datetime(
            raw_index[invalid],
            format="%Y%m%d",
            errors="coerce"
        )
        date_index = pd.DatetimeIndex(date_index)
        date_index = date_index.where(
            ~invalid,
            parsed
        )
    df.index = pd.DatetimeIndex(date_index)
    df.index.name = "Date"
    df = df.loc[~df.index.isna()]
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()
def convert_columns_to_numeric(df):
    """Convert price columns to numeric values."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )
    return df
# ============================================================
# 2. LOAD STOCK DATA
# ============================================================
stock_frames = {}
for stock_name, filename in FILES.items():
    path = os.path.join(DATA_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"\nCSV file not found: {path}\n"
            "Check DATA_DIR and FILES."
        )
    stock_df = read_csv_korean(path)
    stock_df = normalize_date_index(stock_df)
    stock_df = convert_columns_to_numeric(stock_df)
    stock_df.columns = [
        f"{stock_name}_{str(col).strip()}"
        for col in stock_df.columns
    ]
    stock_df = stock_df.dropna(
        axis=1,
        how="all"
    )
    stock_frames[stock_name] = stock_df
    print(
        f"{stock_name}: "
        f"{stock_df.shape}, "
        f"{stock_df.index.min().date()} "
        f"~ {stock_df.index.max().date()}"
    )
# ============================================================
# 3. ALIGN TO SAMSUNG TRADING CALENDAR
# ============================================================
# Samsung's actual trading dates define the master calendar.
# This prevents missing observations in other stocks from
# silently changing the definition of the next Samsung session.
samsung_df = stock_frames["삼성전자"]
if samsung_df.empty:
    raise ValueError("Samsung Electronics data is empty.")
master_dates = samsung_df.index
aligned_frames = []
for stock_name, stock_df in stock_frames.items():
    aligned = stock_df.reindex(master_dates)
    # Forward-fill only previously available observations.
    # No backward fill: future values must not fill past rows.
    aligned = aligned.ffill()
    aligned_frames.append(aligned)
df = pd.concat(
    aligned_frames,
    axis=1
).sort_index()
if TARGET_COL not in df.columns:
    raise KeyError(
        f"Missing target column: {TARGET_COL}\n"
        f"Available columns: {df.columns.tolist()}"
    )
if BASELINE_COL not in df.columns:
    raise KeyError(
        f"Missing baseline column: {BASELINE_COL}\n"
        f"Available columns: {df.columns.tolist()}"
    )
print("\nAligned dataset:")
print("Shape:", df.shape)
print("Date range:", df.index.min(), "~", df.index.max())
# ============================================================
# 4. GENERATE TARGET AND TARGET DATE
# ============================================================
# At each Samsung trading date t:
#   Features = information available at date t
#   Target = Samsung opening price at the next Samsung session
#   Target_Date = the actual next Samsung trading date
df["Target"] = df[TARGET_COL].shift(-1)
df["Target_Return"] = df["Target"] / df[BASELINE_COL] - 1.0
df["Target_Date"] = pd.Series(
    df.index,
    index=df.index
).shift(-1)
# Exclude target and target date from model inputs.
FEATURE_COLS = [
    col for col in df.columns
    if col not in ["Target", "Target_Return", "Target_Date"]
]
# Remaining missing feature values are not imputed from the future.
df[FEATURE_COLS] = df[FEATURE_COLS].ffill()
# Keep a separate full feature frame for forecasting the next session.
# The last row has no known target yet, but is useful for inference.
forecast_features = df[FEATURE_COLS].dropna().copy()
# Labeled rows are used for training and evaluation.
data = df.dropna(
    subset=FEATURE_COLS + ["Target", "Target_Return", "Target_Date"]
).copy()
if len(data) <= TIMESTEPS + 10:
    raise ValueError(
        f"Insufficient labeled data: {len(data)} rows."
    )
if len(forecast_features) < TIMESTEPS:
    raise ValueError(
        f"Need at least {TIMESTEPS} rows for latest forecast."
    )
print("\nTarget configuration:")
print("Target:", TARGET_COL)
print("Model target: next-session open / current-session close - 1 (return)")
print("Baseline:", BASELINE_COL)
print("Number of features:", len(FEATURE_COLS))
print("Labeled rows:", len(data))
print("Latest labeled input date:", data.index[-1])
print("Latest target date:", data["Target_Date"].iloc[-1])
print("Latest available feature date:", forecast_features.index[-1])
# ============================================================
# 5. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================
n = len(data)
train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
if train_end <= TIMESTEPS:
    raise ValueError(
        "Training data is shorter than the required sequence length."
    )
if val_end >= n:
    raise ValueError("Invalid validation/test split.")
print("\nChronological split:")
print(f"Train:      rows 0 to {train_end - 1}")
print(f"Validation: rows {train_end} to {val_end - 1}")
print(f"Test:       rows {val_end} to {n - 1}")
X_raw = data[FEATURE_COLS].to_numpy(
    dtype=np.float32
)
y_raw = data["Target_Return"].to_numpy(
    dtype=np.float32
).reshape(-1, 1)
# ============================================================
# 6. TRAIN-ONLY SCALING
# ============================================================
x_scaler = StandardScaler()
y_scaler = StandardScaler()
# Fit using training rows only.
x_scaler.fit(X_raw[:train_end])
y_scaler.fit(y_raw[:train_end])
# Transform all labeled rows with the training-fitted scalers.
X_scaled = x_scaler.transform(
    X_raw
).astype(np.float32)
y_scaled = y_scaler.transform(
    y_raw
).astype(np.float32)
print("\nScaling completed.")
print("Feature scaler fitted on training rows only.")
print("Target scaler fitted on training rows only.")
# ============================================================
# 7. TIME-SERIES SEQUENCE GENERATION
# ============================================================
def make_sequences(X, y, timesteps):
    """
    Input:
        X[i-timesteps+1 : i+1]
    Label:
        y[i], which represents the next Samsung session's open.
    The input sequence ends at the current input date.
    """
    X_seq = []
    y_seq = []
    positions = []
    for i in range(timesteps - 1, len(X)):
        start = i - timesteps + 1
        end = i + 1
        X_seq.append(X[start:end])
        y_seq.append(y[i])
        positions.append(i)
    return (
        np.asarray(X_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32),
        np.asarray(positions, dtype=np.int64)
    )
X_seq, y_seq, positions = make_sequences(
    X_scaled,
    y_scaled,
    TIMESTEPS
)
# Validation and test sequences may use historical observations
# from the immediately preceding period. This is valid for rolling
# forecasting because those observations were already available.
train_mask = positions < train_end
val_mask = (
    (positions >= train_end)
    & (positions < val_end)
)
test_mask = positions >= val_end
X_train = X_seq[train_mask]
y_train = y_seq[train_mask]
X_val = X_seq[val_mask]
y_val = y_seq[val_mask]
X_test = X_seq[test_mask]
y_test = y_seq[test_mask]
train_positions = positions[train_mask]
val_positions = positions[val_mask]
test_positions = positions[test_mask]
if min(len(X_train), len(X_val), len(X_test)) == 0:
    raise ValueError(
        "At least one split contains no sequences."
    )
print("\nSequence shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:  ", X_val.shape, "y_val:  ", y_val.shape)
print("X_test: ", X_test.shape, "y_test: ", y_test.shape)
# ============================================================
# 8. MODEL 1: LSTM
# ============================================================
def build_lstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ], name="LSTM_Stock_Model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="mse",
        metrics=["mae"]
    )
    return model
# ============================================================
# 9. MODEL 2: Conv1D
# ============================================================
def build_conv1d(input_shape):
    model = Sequential([
        Input(shape=input_shape),
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
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="mse",
        metrics=["mae"]
    )
    return model
# ============================================================
# 10. TRANSFORMER ENCODER BLOCK
# ============================================================
def transformer_encoder(
    x,
    d_model=64,
    num_heads=4,
    ff_dim=128,
    dropout=0.1
):
    # Multi-Head Self-Attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout
    )(x, x)
    # Residual connection and normalization
    x = Add()([
        x,
        Dropout(dropout)(attention_output)
    ])
    x = LayerNormalization(
        epsilon=1e-6
    )(x)
    # Position-wise Feed-Forward Network
    ffn = Dense(
        ff_dim,
        activation="relu"
    )(x)
    ffn = Dropout(dropout)(ffn)
    ffn = Dense(d_model)(ffn)
    # Residual connection and normalization
    x = Add()([
        x,
        Dropout(dropout)(ffn)
    ])
    x = LayerNormalization(
        epsilon=1e-6
    )(x)
    return x
# ============================================================
# 11. MODEL 3: TRANSFORMER
# ============================================================
class AddPositionEmbedding(tf.keras.layers.Layer):
    """Serializable positional embedding addition compatible with older tf.keras."""
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.position_embedding = Embedding(
            input_dim=self.seq_len,
            output_dim=self.d_model,
            name="position_embedding"
        )

    def call(self, inputs):
        # Shape: (seq_len, d_model) -> (1, seq_len, d_model).
        positions = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)
        pos = self.position_embedding(positions)
        pos = tf.expand_dims(pos, axis=0)
        return inputs + pos

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"seq_len": self.seq_len, "d_model": self.d_model})
        return config


def build_transformer(
    input_shape,
    d_model=64,
    num_heads=4,
    ff_dim=128,
    num_blocks=2,
    dropout=0.1
):
    seq_len, num_features = input_shape
    inputs = Input(
        shape=input_shape,
        name="stock_sequence"
    )
    # Project input features to Transformer embedding dimension.
    x = Dense(
        d_model,
        name="input_projection"
    )(inputs)
    # Add positions inside a serializable custom layer. This avoids the old
    # Keras Add rank/broadcast issue and avoids Lambda output-shape serialization.
    x = AddPositionEmbedding(
        seq_len=seq_len,
        d_model=d_model,
        name="add_position_embedding"
    )(x)
    # Transformer Encoder stack.
    for block_idx in range(num_blocks):
        x = transformer_encoder(
            x,
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        )
    # Aggregate temporal representations.
    x = GlobalAveragePooling1D()(x)
    x = Dense(
        64,
        activation="relu"
    )(x)
    x = Dropout(dropout)(x)
    outputs = Dense(
        1,
        name="next_open_scaled"
    )(x)
    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="Transformer_Stock_Model"
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=3e-4
        ),
        loss="mse",
        metrics=["mae"]
    )
    return model
# ============================================================
# 12. BUILD ALL THREE MODELS
# ============================================================
input_shape = (
    TIMESTEPS,
    X_train.shape[2]
)
models = {
    "LSTM": build_lstm(input_shape),
    "Conv1D": build_conv1d(input_shape),
    "Transformer": build_transformer(input_shape)
}

print("\n========== TRANSFORMER SHAPE CHECK ==========")
transformer_model = models["Transformer"]
print("X_train shape:", X_train.shape)
print("Model input shape:", transformer_model.input_shape)
print("Model output shape:", transformer_model.output_shape)
assert X_train.ndim == 3, f"Expected 3D input, got {X_train.shape}"
assert X_train.shape[1] == TIMESTEPS, (
    f"Sequence length mismatch: {X_train.shape[1]} != {TIMESTEPS}"
)
assert X_train.shape[2] == len(FEATURE_COLS), (
    f"Feature count mismatch: {X_train.shape[2]} != {len(FEATURE_COLS)}"
)
assert y_train.ndim == 2 and y_train.shape[1] == 1, (
    f"Unexpected target shape: {y_train.shape}"
)
print("Transformer input validation: PASS")
print("\n========== MODEL PARAMETER COUNTS ==========")
for name, model in models.items():
    print(f"\n{name}: {model.count_params():,} parameters")
    model.summary()
# ============================================================
# 13. TRAIN MODELS
# ============================================================
histories = {}
for name, model in models.items():
    print(f"\n{'=' * 20} TRAINING {name} {'=' * 20}")
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    histories[name] = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        shuffle=False,
        verbose=1
    )
# ============================================================
# 14. PREDICTION AND INVERSE SCALING
# ============================================================
def predict_price(model, X, baseline_close, scaler=None):
    """
    Predict next-session opening price by predicting the return
    relative to the input date's close, then converting back to KRW.
    baseline_close may be a scalar or one value per sequence.
    """
    if scaler is None:
        scaler = y_scaler
    pred_scaled = model.predict(X, verbose=0)
    pred_return = scaler.inverse_transform(pred_scaled).reshape(-1)
    close_values = np.asarray(baseline_close, dtype=np.float64).reshape(-1)
    if close_values.size == 1 and pred_return.size > 1:
        close_values = np.repeat(close_values, pred_return.size)
    if close_values.size != pred_return.size:
        raise ValueError(
            f"baseline_close count ({close_values.size}) does not match "
            f"prediction count ({pred_return.size})"
        )
    return close_values * (1.0 + pred_return)

# Evaluate all predictions in original KRW price space.
y_val_actual = data["Target"].to_numpy(dtype=np.float32)[val_positions]
y_test_actual = data["Target"].to_numpy(dtype=np.float32)[test_positions]
baseline_all = data[BASELINE_COL].to_numpy(dtype=np.float32)
baseline_val = baseline_all[val_positions]
baseline_test = baseline_all[test_positions]

val_predictions = {}
test_predictions = {}
for name, model in models.items():
    val_predictions[name] = predict_price(
        model, X_val, baseline_val
    )
    test_predictions[name] = predict_price(
        model, X_test, baseline_test
    )
# ============================================================
# 15. VALIDATION-SELECTED ENSEMBLE WITH BASELINE FALLBACK
# ============================================================
# Search weights using validation MAE only. The baseline is included
# so the selected ensemble can fall back to the current-close forecast
# when neural models do not generalize. Test labels are never used here.
weight_grid = np.linspace(0.0, 1.0, 21)
best_weights = None
best_val_mae = float("inf")
weight_results = []

for w_lstm in weight_grid:
    for w_conv in weight_grid:
        for w_transformer in weight_grid:
            w_baseline = 1.0 - w_lstm - w_conv - w_transformer
            if w_baseline < -1e-9:
                continue
            w_baseline = max(0.0, w_baseline)
            ensemble_val = (
                w_lstm * val_predictions["LSTM"]
                + w_conv * val_predictions["Conv1D"]
                + w_transformer * val_predictions["Transformer"]
                + w_baseline * baseline_val
            )
            val_mae = mean_absolute_error(y_val_actual, ensemble_val)
            weight_results.append({
                "LSTM_Weight": float(w_lstm),
                "Conv1D_Weight": float(w_conv),
                "Transformer_Weight": float(w_transformer),
                "Baseline_Weight": float(w_baseline),
                "Validation_MAE": float(val_mae)
            })
            if val_mae < best_val_mae:
                best_val_mae = float(val_mae)
                best_weights = {
                    "LSTM": float(w_lstm),
                    "Conv1D": float(w_conv),
                    "Transformer": float(w_transformer),
                    "Baseline_Current_Close": float(w_baseline)
                }

ensemble_test = (
    best_weights["LSTM"] * test_predictions["LSTM"]
    + best_weights["Conv1D"] * test_predictions["Conv1D"]
    + best_weights["Transformer"] * test_predictions["Transformer"]
    + best_weights["Baseline_Current_Close"] * baseline_test
)
print("\n========== ENSEMBLE WEIGHTS ==========")
print(json.dumps(
    best_weights,
    indent=2,
    ensure_ascii=False
))
print(f"Best Validation MAE: {best_val_mae:,.4f}")
# ============================================================
# 16. TEST METRICS
# ============================================================
def calculate_metrics(y_true, y_pred, model_name):
    return {
        "Model": model_name,
        "MAE": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(y_true, y_pred)
            )
        )
    }
# Baseline: current Samsung close predicts next session's open.
metrics = []
for name in models:
    metrics.append(
        calculate_metrics(
            y_test_actual,
            test_predictions[name],
            name
        )
    )
metrics.append(
    calculate_metrics(
        y_test_actual,
        ensemble_test,
        "Ensemble_3Models_Plus_Baseline"
    )
)
metrics.append(
    calculate_metrics(
        y_test_actual,
        baseline_test,
        "Baseline_Current_Close"
    )
)
metrics_df = pd.DataFrame(
    metrics
).sort_values("MAE")
print("\n========== TEST METRICS ==========")
print(metrics_df.to_string(index=False))
metrics_df.to_csv(
    os.path.join(SAVE_DIR, "test_metrics_v3_2.csv"),
    index=False,
    encoding="utf-8-sig"
)
# ============================================================
# 17. SAVE TEST PREDICTIONS
# ============================================================
test_dates = pd.to_datetime(
    data["Target_Date"].iloc[test_positions]
).to_numpy()
predictions_df = pd.DataFrame({
    "Input_Date": data.index[test_positions],
    "Target_Date": test_dates,
    "Actual_Next_Open": y_test_actual,
    "Baseline_Current_Close": baseline_test,
    "LSTM_Prediction": test_predictions["LSTM"],
    "Conv1D_Prediction": test_predictions["Conv1D"],
    "Transformer_Prediction": test_predictions["Transformer"],
    "Ensemble_3Models_Plus_Baseline": ensemble_test
})
predictions_df.to_csv(
    os.path.join(SAVE_DIR, "test_predictions_v3_2.csv"),
    index=False,
    encoding="utf-8-sig"
)
pd.DataFrame(weight_results).to_csv(
    os.path.join(SAVE_DIR, "ensemble_weights_v3_2.csv"),
    index=False,
    encoding="utf-8-sig"
)
print("\nTest predictions:")
print(predictions_df.tail(10).to_string(index=False))
# ============================================================
# 18. FORECAST NEXT SAMSUNG TRADING SESSION
# ============================================================
# Use the latest available feature date, including the latest row
# that does not yet have a known next-session target.
latest_features = forecast_features[
    FEATURE_COLS
].to_numpy(dtype=np.float32)
if len(latest_features) < TIMESTEPS:
    raise ValueError(
        "Insufficient latest feature rows for forecasting."
    )
latest_window_raw = latest_features[-TIMESTEPS:]
latest_window_scaled = x_scaler.transform(
    latest_window_raw
).astype(np.float32)
latest_window = np.expand_dims(
    latest_window_scaled,
    axis=0
)
latest_baseline_close = float(forecast_features[BASELINE_COL].iloc[-1])
latest_predictions = {
    name: float(
        predict_price(model, latest_window, latest_baseline_close)[0]
    )
    for name, model in models.items()
}
latest_ensemble = (
    best_weights["LSTM"] * latest_predictions["LSTM"]
    + best_weights["Conv1D"] * latest_predictions["Conv1D"]
    + best_weights["Transformer"] * latest_predictions["Transformer"]
    + best_weights["Baseline_Current_Close"] * latest_baseline_close
)
last_input_date = forecast_features.index[-1]
next_prediction_df = pd.DataFrame([{
    "Last_Input_Date": last_input_date,
    "LSTM_Next_Open": latest_predictions["LSTM"],
    "Conv1D_Next_Open": latest_predictions["Conv1D"],
    "Transformer_Next_Open": latest_predictions["Transformer"],
    "Ensemble_Next_Open": latest_ensemble,
    "LSTM_Weight": best_weights["LSTM"],
    "Conv1D_Weight": best_weights["Conv1D"],
    "Transformer_Weight": best_weights["Transformer"],
    "Baseline_Weight": best_weights["Baseline_Current_Close"],
    "Current_Close_Baseline": latest_baseline_close,
    "Note": "Model estimate; not a guaranteed market price"
}])
next_prediction_df.to_csv(
    os.path.join(SAVE_DIR, "next_prediction_v3_2.csv"),
    index=False,
    encoding="utf-8-sig"
)
print("\n========== NEXT SESSION OPEN FORECAST ==========")
print(next_prediction_df.to_string(index=False))
# ============================================================
# 19. SAVE MODELS, SCALERS AND CONFIG
# ============================================================
for name, model in models.items():
    model_path = os.path.join(
        SAVE_DIR,
        f"{name.lower()}_model_v3_2.keras"
    )
    model.save(model_path)
    pd.DataFrame(
        histories[name].history
    ).to_csv(
        os.path.join(
            SAVE_DIR,
            f"history_{name.lower()}_v3_2.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )
joblib.dump(
    x_scaler,
    os.path.join(SAVE_DIR, "x_scaler_v3_2.pkl")
)
joblib.dump(
    y_scaler,
    os.path.join(SAVE_DIR, "y_scaler_v3_2.pkl")
)
config = {
    "project": "CRAZY AI Stock Prediction V3.2",
    "target": TARGET_COL,
    "model_target": "next_open_divided_by_current_close_minus_one",
    "baseline": BASELINE_COL,
    "timesteps": TIMESTEPS,
    "train_ratio": TRAIN_RATIO,
    "validation_ratio": VAL_RATIO,
    "test_ratio": TEST_RATIO,
    "feature_columns": FEATURE_COLS,
    "ensemble_weights": best_weights,
    "best_validation_mae": best_val_mae,
    "model_parameters": {
        name: int(model.count_params())
        for name, model in models.items()
    },
    "seed": SEED
}
with open(
    os.path.join(SAVE_DIR, "config_v3_2.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        config,
        f,
        ensure_ascii=False,
        indent=2
    )
# ============================================================
# 20. RELOAD AND VERIFY SAVED MODELS
# ============================================================
reloaded_x_scaler = joblib.load(
    os.path.join(SAVE_DIR, "x_scaler_v3_2.pkl")
)
reloaded_y_scaler = joblib.load(
    os.path.join(SAVE_DIR, "y_scaler_v3_2.pkl")
)
reload_results = {}
for name in models:
    model_path = os.path.join(
        SAVE_DIR,
        f"{name.lower()}_model_v3_2.keras"
    )
    # compile=False avoids optimizer deserialization issues during inference.
    try:
        reloaded_model = tf.keras.models.load_model(
            model_path,
            custom_objects={"AddPositionEmbedding": AddPositionEmbedding},
            compile=False
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to reload {name} from {model_path}. "
            "Delete old *_model_v3.keras files created by previous versions, "
            "then rerun the entire script so all models are rebuilt and saved "
            "with the corrected architecture. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    pred_after_reload = predict_price(
        reloaded_model,
        X_test,
        baseline_test,
        reloaded_y_scaler
    )
    pred_before_reload = test_predictions[name]
    reload_results[name] = bool(
        np.allclose(
            pred_before_reload,
            pred_after_reload,
            rtol=1e-5,
            atol=1e-4
        )
    )
# Verify scaler outputs as well.
reload_results["X_Scaler"] = bool(
    np.allclose(
        x_scaler.transform(X_raw[-TIMESTEPS:]),
        reloaded_x_scaler.transform(X_raw[-TIMESTEPS:]),
        rtol=1e-6,
        atol=1e-6
    )
)
reload_results["Y_Scaler"] = bool(
    np.allclose(
        y_scaler.transform(y_raw[-TIMESTEPS:]),
        reloaded_y_scaler.transform(y_raw[-TIMESTEPS:]),
        rtol=1e-6,
        atol=1e-6
    )
)
reload_results["Overall_Pass"] = all(
    reload_results.values()
)
with open(
    os.path.join(SAVE_DIR, "reload_verification_v3_2.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        reload_results,
        f,
        ensure_ascii=False,
        indent=2
    )
print("\n========== RELOAD VERIFICATION ==========")
print(json.dumps(
    reload_results,
    ensure_ascii=False,
    indent=2
))
# ============================================================
# 21. PLOT TRAINING LOSS
# ============================================================
plt.figure(figsize=(12, 6))
for name, history in histories.items():
    plt.plot(
        history.history["loss"],
        label=f"{name} Train"
    )
    plt.plot(
        history.history["val_loss"],
        label=f"{name} Validation"
    )
plt.title("CRAZY AI V3.0 - Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(SAVE_DIR, "training_loss_v3_2.png"),
    dpi=150
)
plt.show()
# ============================================================
# 22. PLOT TEST MAE
# ============================================================
plt.figure(figsize=(12, 6))
plt.bar(
    metrics_df["Model"],
    metrics_df["MAE"]
)
plt.title("Test MAE Comparison")
plt.ylabel("MAE (KRW)")
plt.xticks(rotation=20)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(SAVE_DIR, "test_mae_comparison_v3_2.png"),
    dpi=150
)
plt.show()
# ============================================================
# 23. PLOT TEST PREDICTIONS
# ============================================================
plt.figure(figsize=(14, 6))
plt.plot(
    test_dates,
    y_test_actual,
    label="Actual Next Open"
)
plt.plot(
    test_dates,
    test_predictions["LSTM"],
    label="LSTM"
)
plt.plot(
    test_dates,
    test_predictions["Conv1D"],
    label="Conv1D"
)
plt.plot(
    test_dates,
    test_predictions["Transformer"],
    label="Transformer"
)
plt.plot(
    test_dates,
    ensemble_test,
    label="Ensemble"
)
plt.plot(
    test_dates,
    baseline_test,
    label="Current Close Baseline",
    alpha=0.7
)
plt.title("Samsung Electronics Next Trading Session Open")
plt.xlabel("Target Date")
plt.ylabel("Price (KRW)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(SAVE_DIR, "test_predictions_v3_2.png"),
    dpi=150
)
plt.show()
# ============================================================
# 24. FINAL SUMMARY
# ============================================================
print("\n" + "=" * 65)
print("CRAZY AI STOCK PREDICTION V3.2 COMPLETED")
print("=" * 65)
print("\nModel parameters:")
for name, model in models.items():
    print(f"{name:15s}: {model.count_params():,}")
print("\nTest metrics:")
print(metrics_df.to_string(index=False))
print("\nNext session forecast:")
print(next_prediction_df.to_string(index=False))
print("\nReload verification:")
print(json.dumps(
    reload_results,
    ensure_ascii=False,
    indent=2
))
print("\nSaved files:")
for filename in sorted(os.listdir(SAVE_DIR)):
    path = os.path.join(SAVE_DIR, filename)
    if os.path.isfile(path):
        print(
            f"{filename:40s} "
            f"{os.path.getsize(path):,} bytes"
        )
print("\nOutput directory:", os.path.abspath(SAVE_DIR))