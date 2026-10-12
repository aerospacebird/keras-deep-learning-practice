
# ============================================================
# SAMSUNG STOCK PREDICTION V2.0
# LSTM + Conv1D + Ensemble
# Google Colab / Pandas / Scikit-learn / Keras
#
# Target: 삼성전자 다음 거래일 시가
# Models: LSTM, Conv1D, Weighted Ensemble
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# SAMSUNG STOCK PREDICTION V2.1
# LSTM + Conv1D + Ensemble
# Google Colab / Pandas / Scikit-learn / Keras
#
# Target : 삼성전자 다음 거래일 시가
# Models : LSTM, Conv1D, Weighted Ensemble
# Save   : ./_data/_save/keras70/
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Conv1D, Dropout, Dense, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# 1. 데이터 로드
# ============================================================

path = './_data/samsung/'

def read_csv_korean(file_path):
    for encoding in ['utf-8-sig', 'cp949', 'euc-kr']:
        try:
            return pd.read_csv(file_path, index_col=0, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f'CSV 인코딩을 확인하세요: {file_path}')

df1 = read_csv_korean(path + '아모레261012.csv')
df2 = read_csv_korean(path + '하이닉스261012.csv')
df3 = read_csv_korean(path + '삼성전자261012.csv')

print('아모레:', df1.shape)
print('SK하이닉스:', df2.shape)
print('삼성전자:', df3.shape)

print('\n삼성전자 컬럼:')
print(df3.columns.tolist())

# ============================================================
# 2. 날짜 인덱스 변환
# ============================================================

for d in [df1, df2, df3]:
    date_index = d.index.astype(str).str.strip()
    parsed = pd.to_datetime(date_index, errors='coerce')

    invalid = parsed.isna()

    if invalid.any():
        parsed = pd.Series(parsed, index=d.index)
        parsed.loc[invalid] = pd.to_datetime(
            date_index[invalid], format='%Y%m%d', errors='coerce'
        )
        parsed = pd.DatetimeIndex(parsed)

    d.index = pd.DatetimeIndex(parsed)
    d.drop(index=d.index[d.index.isna()], inplace=True)
    d.sort_index(inplace=True)

# ============================================================
# 3. 컬럼명 구분
# ============================================================

df1.columns = ['아모레_' + str(c).strip() for c in df1.columns]
df2.columns = ['하이닉스_' + str(c).strip() for c in df2.columns]
df3.columns = ['삼성전자_' + str(c).strip() for c in df3.columns]

# ============================================================
# 4. 날짜 기준 데이터 통합
# ============================================================

df = pd.concat([df1, df2, df3], axis=1, join='inner').sort_index()
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna(axis=1, how='all')
df = df.ffill()

print('\n통합 데이터:', df.shape)
print('데이터 기간:', df.index.min(), '~', df.index.max())
print(df.head())

required_cols = ['삼성전자_시가', '삼성전자_종가']

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f'필수 컬럼이 없습니다: {col}')

# ============================================================
# 5. 다음 거래일 시가 Target 생성
# ============================================================

# 현재 날짜의 특성으로 다음 관측일의 시가 예측
df['Target'] = df['삼성전자_시가'].shift(-1)

# Target이 없는 마지막 행 제외
data = df.dropna(subset=['Target']).copy()

# 입력 변수에는 Target을 포함하지 않음
feature_cols = [col for col in df.columns if col != 'Target']

# 입력 특성에 결측치가 남은 행 제거
data = data.dropna(subset=feature_cols)

X = data[feature_cols].values.astype(np.float32)
y = data['Target'].values.astype(np.float32)

# 각 입력 행에 대응하는 예측 대상 날짜 보존
target_date_map = pd.Series(
    df.index[1:],
    index=df.index[:-1]
)

data['Target_Date'] = data.index.map(target_date_map)

print('\nX:', X.shape)
print('y:', y.shape)
print('특성 수:', len(feature_cols))

# ============================================================
# 6. 시간 순서에 따른 데이터 분할
# Train 70% / Validation 15% / Test 15%
# ============================================================

n = len(data)

if n < 100:
    raise ValueError('학습 데이터가 너무 적습니다.')

train_end = int(n * 0.70)
val_end = int(n * 0.85)

print('\nTrain:', train_end)
print('Validation:', val_end - train_end)
print('Test:', n - val_end)

# ============================================================
# 7. 데이터 정규화
# Scaler는 학습 데이터로만 fit
# ============================================================

scaler_X = StandardScaler()
scaler_y = StandardScaler()

scaler_X.fit(X[:train_end])
scaler_y.fit(y[:train_end].reshape(-1, 1))

X_scaled = scaler_X.transform(X).astype(np.float32)
y_scaled = scaler_y.transform(y.reshape(-1, 1)).ravel().astype(np.float32)

# ============================================================
# 8. 시계열 데이터 생성
# 과거 30개 관측일 사용
# ============================================================

TIMESTEPS = 30

X_seq = []
y_seq = []

for i in range(TIMESTEPS, len(X_scaled)):
    X_seq.append(X_scaled[i - TIMESTEPS:i])
    y_seq.append(y_scaled[i])

X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.float32)

# 각 시퀀스 정답의 원본 data 위치
target_positions = np.arange(TIMESTEPS, len(data))

train_mask = target_positions < train_end
val_mask = (target_positions >= train_end) & (target_positions < val_end)
test_mask = target_positions >= val_end

X_train, y_train = X_seq[train_mask], y_seq[train_mask]
X_val, y_val = X_seq[val_mask], y_seq[val_mask]
X_test, y_test = X_seq[test_mask], y_seq[test_mask]

# 실제 주가와 대상 날짜를 동일한 순서로 정렬
y_val_actual = y[target_positions[val_mask]]
y_test_actual = y[target_positions[test_mask]]

val_dates = data['Target_Date'].iloc[target_positions[val_mask]].to_numpy()
test_dates = data['Target_Date'].iloc[target_positions[test_mask]].to_numpy()

print('\n시퀀스 데이터')
print('X_train:', X_train.shape, 'y_train:', y_train.shape)
print('X_val  :', X_val.shape, 'y_val  :', y_val.shape)
print('X_test :', X_test.shape, 'y_test :', y_test.shape)

if min(len(X_train), len(X_val), len(X_test)) == 0:
    raise ValueError('학습·검증·테스트 데이터가 부족합니다.')

# ============================================================
# 9. LSTM MODEL
# 수평형 모델 정의
# ============================================================

lstm_model = Sequential([
    Input(shape=(TIMESTEPS, X_train.shape[2])),
    LSTM(64, return_sequences=True), Dropout(0.2),
    LSTM(32), Dropout(0.2),
    Dense(16, activation='relu'), Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print('\n===== LSTM MODEL =====')
lstm_model.summary()

# ============================================================
# 10. Conv1D MODEL
# 수평형 모델 정의
# ============================================================

conv_model = Sequential([
    Input(shape=(TIMESTEPS, X_train.shape[2])),
    Conv1D(64, 3, padding='causal', activation='relu'), Dropout(0.2),
    Conv1D(32, 3, padding='causal', activation='relu'),
    GlobalAveragePooling1D(),
    Dense(32, activation='relu'), Dropout(0.2), Dense(1)
])

conv_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print('\n===== Conv1D MODEL =====')
conv_model.summary()

print('\nLSTM parameters:', lstm_model.count_params())
print('Conv1D parameters:', conv_model.count_params())

# ============================================================
# 11. 공통 학습 설정
# ============================================================

early_stop = EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)

# ============================================================
# 12. LSTM 학습
# ============================================================

print('\n===== LSTM TRAINING =====')

history_lstm = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100, batch_size=32, shuffle=False,
    callbacks=[early_stop], verbose=1
)

# ============================================================
# 13. Conv1D 학습
# ============================================================

print('\n===== Conv1D TRAINING =====')

history_conv = conv_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100, batch_size=32, shuffle=False,
    callbacks=[early_stop], verbose=1
)

# ============================================================
# 14. 검증 데이터 예측 및 앙상블 가중치 선택
# ============================================================

val_lstm_scaled = lstm_model.predict(X_val, verbose=0)
val_conv_scaled = conv_model.predict(X_val, verbose=0)

val_lstm = scaler_y.inverse_transform(val_lstm_scaled).ravel()
val_conv = scaler_y.inverse_transform(val_conv_scaled).ravel()

# LSTM 가중치 w를 0~1 구간에서 탐색
weights = np.linspace(0.0, 1.0, 101)
ensemble_val_mae = []

for w in weights:
    ensemble_val = w * val_lstm + (1.0 - w) * val_conv
    ensemble_val_mae.append(
        mean_absolute_error(y_val_actual, ensemble_val)
    )

best_idx = int(np.argmin(ensemble_val_mae))
best_w = float(weights[best_idx])
best_conv_w = 1.0 - best_w

print('\n===== ENSEMBLE WEIGHTS =====')
print(f'LSTM weight  : {best_w:.2f}')
print(f'Conv1D weight: {best_conv_w:.2f}')
print(f'Validation MAE: {ensemble_val_mae[best_idx]:,.2f}원')

# ============================================================
# 15. 테스트 데이터 예측
# ============================================================

test_lstm_scaled = lstm_model.predict(X_test, verbose=0)
test_conv_scaled = conv_model.predict(X_test, verbose=0)

test_lstm = scaler_y.inverse_transform(test_lstm_scaled).ravel()
test_conv = scaler_y.inverse_transform(test_conv_scaled).ravel()

test_ensemble = best_w * test_lstm + best_conv_w * test_conv

# ============================================================
# 16. 단순 기준 모델
# 현재 관측일의 종가를 다음 관측일 시가의 예측값으로 사용
# ============================================================

baseline_test = data['삼성전자_종가'].to_numpy()[target_positions[test_mask]]

# ============================================================
# 17. 테스트 성능 평가
# ============================================================

results = []

for name, prediction in [
    ('LSTM', test_lstm),
    ('Conv1D', test_conv),
    ('Ensemble', test_ensemble),
    ('Baseline', baseline_test)
]:
    model_mae = mean_absolute_error(y_test_actual, prediction)
    model_rmse = np.sqrt(mean_squared_error(y_test_actual, prediction))

    results.append({
        'Model': name,
        'MAE': model_mae,
        'RMSE': model_rmse
    })

results_df = pd.DataFrame(results)

print('\n===== TEST PERFORMANCE =====')
display(results_df.round(2))

# ============================================================
# 18. 예측 결과 시각화
# ============================================================

plt.figure(figsize=(15, 6))
plt.plot(y_test_actual, label='Actual Open')
plt.plot(test_lstm, label='LSTM')
plt.plot(test_conv, label='Conv1D')
plt.plot(test_ensemble, label='Ensemble')
plt.plot(baseline_test, label='Baseline', alpha=0.7)
plt.title('Samsung Next-Day Open Prediction')
plt.xlabel('Test Sample')
plt.ylabel('Price (KRW)')
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# 19. 학습 곡선 비교
# ============================================================

plt.figure(figsize=(12, 5))
plt.plot(history_lstm.history['val_loss'], label='LSTM Validation Loss')
plt.plot(history_conv.history['val_loss'], label='Conv1D Validation Loss')
plt.title('Validation Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# 20. 다음 거래일 시가 예측
# ============================================================

latest_data = df[feature_cols].ffill().tail(TIMESTEPS)

if len(latest_data) != TIMESTEPS:
    raise ValueError('최신 데이터 30개가 부족합니다.')

latest_scaled = scaler_X.transform(latest_data.values).astype(np.float32)
latest_scaled = latest_scaled.reshape(1, TIMESTEPS, len(feature_cols))

next_lstm_scaled = lstm_model.predict(latest_scaled, verbose=0)
next_conv_scaled = conv_model.predict(latest_scaled, verbose=0)

next_lstm = scaler_y.inverse_transform(next_lstm_scaled)[0, 0]
next_conv = scaler_y.inverse_transform(next_conv_scaled)[0, 0]
next_ensemble = best_w * next_lstm + best_conv_w * next_conv

print('\n===== NEXT TRADING DAY PREDICTION =====')
print('최종 관측일:', df.index[-1].strftime('%Y-%m-%d'))
print(f'LSTM 예측 시가    : {next_lstm:,.0f}원')
print(f'Conv1D 예측 시가  : {next_conv:,.0f}원')
print(f'Ensemble 예측 시가: {next_ensemble:,.0f}원')

# ============================================================
# 21. 모델 및 가중치 저장
# 요청한 저장 경로로 통일
# ============================================================

save_path = './_data/_save/keras70/'
os.makedirs(save_path, exist_ok=True)

# 전체 모델 저장
lstm_model.save(save_path + 'samsung_lstm_model.keras')
conv_model.save(save_path + 'samsung_conv1d_model.keras')

# 가중치만 별도 저장
lstm_model.save_weights(save_path + 'samsung_lstm_weights.weights.h5')
conv_model.save_weights(save_path + 'samsung_conv1d_weights.weights.h5')

# Scaler 저장
joblib.dump(scaler_X, save_path + 'scaler_X.pkl')
joblib.dump(scaler_y, save_path + 'scaler_y.pkl')

# 모델 설정 저장
config = {
    'TIMESTEPS': TIMESTEPS,
    'feature_cols': feature_cols,
    'n_features': len(feature_cols),
    'target': '삼성전자 다음 거래일 시가',
    'lstm_weight': best_w,
    'conv1d_weight': best_conv_w,
    'last_observation_date': df.index[-1].strftime('%Y-%m-%d')
}

with open(save_path + 'model_config.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=4)

# 성능 지표 저장
results_df.to_csv(
    save_path + 'test_metrics.csv', index=False, encoding='utf-8-sig'
)

# 앙상블 가중치 탐색 결과 저장
pd.DataFrame({
    'LSTM_Weight': weights,
    'Conv1D_Weight': 1.0 - weights,
    'Validation_MAE': ensemble_val_mae
}).to_csv(
    save_path + 'ensemble_weight_search.csv',
    index=False, encoding='utf-8-sig'
)

# 학습 기록 저장
pd.DataFrame(history_lstm.history).to_csv(
    save_path + 'lstm_history.csv', index=False, encoding='utf-8-sig'
)

pd.DataFrame(history_conv.history).to_csv(
    save_path + 'conv1d_history.csv', index=False, encoding='utf-8-sig'
)

# 테스트 예측 저장
prediction_df = pd.DataFrame({
    'Target_Date': test_dates,
    'Actual_Open': y_test_actual,
    'LSTM_Prediction': test_lstm,
    'Conv1D_Prediction': test_conv,
    'Ensemble_Prediction': test_ensemble,
    'Baseline_Prediction': baseline_test
})

prediction_df.to_csv(
    save_path + 'test_predictions.csv',
    index=False, encoding='utf-8-sig'
)

# 다음 거래일 예측 저장
pd.DataFrame([{
    'Last_Observation_Date': df.index[-1].strftime('%Y-%m-%d'),
    'LSTM_Prediction': next_lstm,
    'Conv1D_Prediction': next_conv,
    'Ensemble_Prediction': next_ensemble,
    'LSTM_Weight': best_w,
    'Conv1D_Weight': best_conv_w
}]).to_csv(
    save_path + 'next_prediction.csv',
    index=False, encoding='utf-8-sig'
)

print('\n저장 완료:', save_path)

# ============================================================
# 22. 저장 모델 불러오기 및 예측 검증
# ============================================================

loaded_lstm = load_model(save_path + 'samsung_lstm_model.keras')
loaded_conv = load_model(save_path + 'samsung_conv1d_model.keras')

loaded_scaler_X = joblib.load(save_path + 'scaler_X.pkl')
loaded_scaler_y = joblib.load(save_path + 'scaler_y.pkl')

original_lstm_check = lstm_model.predict(X_test[:5], verbose=0)
loaded_lstm_check = loaded_lstm.predict(X_test[:5], verbose=0)

original_conv_check = conv_model.predict(X_test[:5], verbose=0)
loaded_conv_check = loaded_conv.predict(X_test[:5], verbose=0)

print('\n===== MODEL RESTORE CHECK =====')

print('LSTM 복원 검증:',
      np.allclose(original_lstm_check, loaded_lstm_check, rtol=1e-5, atol=1e-6))

print('Conv1D 복원 검증:',
      np.allclose(original_conv_check, loaded_conv_check, rtol=1e-5, atol=1e-6))

# ============================================================
# 23. 별도 가중치 파일 복원 검증
# ============================================================

restored_lstm = Sequential([
    Input(shape=(TIMESTEPS, X_train.shape[2])),
    LSTM(64, return_sequences=True), Dropout(0.2),
    LSTM(32), Dropout(0.2),
    Dense(16, activation='relu'), Dense(1)
])

restored_lstm.compile(optimizer='adam', loss='mse', metrics=['mae'])
restored_lstm.load_weights(save_path + 'samsung_lstm_weights.weights.h5')

restored_conv = Sequential([
    Input(shape=(TIMESTEPS, X_train.shape[2])),
    Conv1D(64, 3, padding='causal', activation='relu'), Dropout(0.2),
    Conv1D(32, 3, padding='causal', activation='relu'),
    GlobalAveragePooling1D(),
    Dense(32, activation='relu'), Dropout(0.2), Dense(1)
])

restored_conv.compile(optimizer='adam', loss='mse', metrics=['mae'])
restored_conv.load_weights(save_path + 'samsung_conv1d_weights.weights.h5')

print('LSTM 가중치 복원 검증:',
      np.allclose(
          lstm_model.predict(X_test[:5], verbose=0),
          restored_lstm.predict(X_test[:5], verbose=0),
          rtol=1e-5, atol=1e-6
      ))

print('Conv1D 가중치 복원 검증:',
      np.allclose(
          conv_model.predict(X_test[:5], verbose=0),
          restored_conv.predict(X_test[:5], verbose=0),
          rtol=1e-5, atol=1e-6
      ))

print('\nCRAZY AI SAMSUNG STOCK PREDICTION V2.1 완료')
print('최종 저장 경로:', save_path)

