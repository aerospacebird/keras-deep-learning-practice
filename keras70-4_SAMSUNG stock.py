
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
from tensorflow.keras.layers import LSTM, Conv1D, Dropout, Dense, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# 1. 데이터
# ============================================================
path = './_data/samsung/'

def read_csv_korean(file_path):
    for encoding in ['utf-8-sig', 'cp949', 'euc-kr']:
        try:
            return pd.read_csv(file_path, index_col=0, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f'CSV 인코딩 확인 필요: {file_path}')

df1 = read_csv_korean(path + '아모레261012.csv')
df2 = read_csv_korean(path + '하이닉스261012.csv')
df3 = read_csv_korean(path + '삼성전자261012.csv')

print('아모레:', df1.shape)
print('SK하이닉스:', df2.shape)
print('삼성전자:', df3.shape)
print('삼성전자 컬럼:', df3.columns.tolist())

print('===== 원본 컬럼명 확인 =====')
print('아모레 컬럼:', df1.columns.tolist())
print('하이닉스 컬럼:', df2.columns.tolist())
print('삼성전자 컬럼:', df3.columns.tolist())
#exit()
# ============================================================
# 2. 날짜 인덱스 처리
# ============================================================
for d in [df1, df2, df3]:
    date_index = d.index.astype(str).str.strip()
    parsed = pd.to_datetime(date_index, errors='coerce')
    
    invalid = parsed.isna()
    if invalid.any():
        parsed = pd.Series(parsed, index=d.index)
        parsed.loc[invalid] = pd.to_datetime(date_index[invalid], format='%Y%m%d', errors='coerce')
        parsed = pd.DatetimeIndex(parsed)
    
    d.index = pd.DatetimeIndex(parsed)
    d.drop(index=d.index[d.index.isna()], inplace=True)
    d.sort_index(inplace=True)

# ============================================================
# 3. 컬럼명 구분 + 데이터 결합 (강제 버전)
# ============================================================

# Unnamed 컬럼 제거
df1 = df1.loc[:, ~df1.columns.astype(str).str.contains('Unnamed')]
df2 = df2.loc[:, ~df2.columns.astype(str).str.contains('Unnamed')]
df3 = df3.loc[:, ~df3.columns.astype(str).str.contains('Unnamed')]

print('Unnamed 제거 후 컬럼 수 → 아모레:', df1.shape[1], '하이닉스:', df2.shape[1], '삼성:', df3.shape[1])

# 컬럼 개수가 15개인지 확인
print('삼성전자 현재 컬럼:', df3.columns.tolist())

# 강제로 컬럼명 지정 (가장 확실한 방법)
base_cols = ['시가', '고가', '저가', '종가', '대비', '등락률', '거래량',
             '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비']

df1.columns = ['아모레_' + c for c in base_cols]
df2.columns = ['하이닉스_' + c for c in base_cols]
df3.columns = ['삼성전자_' + c for c in base_cols]

print('\n강제 지정 후 삼성전자 컬럼:')
print(df3.columns.tolist())
print('삼성전자_시가 존재 여부:', '삼성전자_시가' in df3.columns)

# 결합
df = pd.concat([df1, df2, df3], axis=1, join='inner').sort_index()

print('\nconcat 후 삼성전자_시가 존재 여부:', '삼성전자_시가' in df.columns)

# 그래도 없으면 에러 발생시키기
if '삼성전자_시가' not in df.columns:
    print('\n현재 df 컬럼 전체:')
    print(df.columns.tolist())
    raise ValueError('삼성전자_시가 컬럼이 없습니다. 위 출력을 확인하세요.')

df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna(axis=1, how='all')
df = df.ffill()

print('\n통합 데이터:', df.shape)
print('데이터 기간:', df.index.min(), '~', df.index.max())

# ============================================================
# 4. 다음 거래일 시가 정답 생성
# ============================================================
df['Target'] = df['삼성전자_시가'].shift(-1)
data = df.dropna(subset=['Target']).copy()

feature_cols = [col for col in df.columns if col != 'Target']
data = data.dropna(subset=feature_cols)

X = data[feature_cols].values.astype(np.float32)
y = data['Target'].values.astype(np.float32)

print('\nX:', X.shape)
print('y:', y.shape)
print('Feature 수:', len(feature_cols))

# ============================================================
# 5. 시간 순서 분할 (Train 70% / Val 15% / Test 15%)
# ============================================================
n = len(data)
if n < 100:
    raise ValueError('데이터가 너무 적습니다.')

train_end = int(n * 0.70)
val_end   = int(n * 0.85)

# ============================================================
# 6. Scaler (학습 구간에만 fit)
# ============================================================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

scaler_X.fit(X[:train_end])
scaler_y.fit(y[:train_end].reshape(-1, 1))

X_scaled = scaler_X.transform(X).astype(np.float32)
y_scaled = scaler_y.transform(y.reshape(-1, 1)).ravel().astype(np.float32)

# ============================================================
# 7. 시계열 데이터 생성 (timesteps=30)
# ============================================================
TIMESTEPS = 30

X_seq = []
y_seq = []

for i in range(TIMESTEPS, len(X_scaled)):
    X_seq.append(X_scaled[i - TIMESTEPS : i])
    y_seq.append(y_scaled[i])

X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.float32)

target_positions = np.arange(TIMESTEPS, len(data))

train_mask = target_positions < train_end
val_mask   = (target_positions >= train_end) & (target_positions < val_end)
test_mask  = target_positions >= val_end

X_train, y_train = X_seq[train_mask], y_seq[train_mask]
X_val,   y_val   = X_seq[val_mask],   y_seq[val_mask]
X_test,  y_test  = X_seq[test_mask],  y_seq[test_mask]

print('\n시퀀스 데이터')
print('Train:', X_train.shape, y_train.shape)
print('Val  :', X_val.shape, y_val.shape)
print('Test :', X_test.shape, y_test.shape)

y_val_actual  = y[target_positions[val_mask]]
y_test_actual = y[target_positions[test_mask]]
test_target_dates = df.index[target_positions[test_mask] + 1]

# ============================================================
# 8. LSTM 모델
# ============================================================
lstm_model = Sequential()
lstm_model.add(LSTM(64, return_sequences=True, input_shape=(TIMESTEPS, X_train.shape[2])))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(32))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(16, activation='relu'))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print('\n===== LSTM MODEL =====')
lstm_model.summary()

# ============================================================
# 9. Conv1D 모델
# ============================================================
conv_model = Sequential()
conv_model.add(Conv1D(64, kernel_size=3, padding='causal', activation='relu',
                      input_shape=(TIMESTEPS, X_train.shape[2])))
conv_model.add(Dropout(0.2))
conv_model.add(Conv1D(32, kernel_size=3, padding='causal', activation='relu'))
conv_model.add(GlobalAveragePooling1D())
conv_model.add(Dense(32, activation='relu'))
conv_model.add(Dropout(0.2))
conv_model.add(Dense(1))

conv_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print('\n===== Conv1D MODEL =====')
conv_model.summary()

# ============================================================
# 10. 컴파일, 훈련
# ============================================================
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print('\n===== LSTM TRAINING =====')
history_lstm = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    shuffle=False,
    callbacks=[es],
    verbose=1
)

print('\n===== Conv1D TRAINING =====')
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
# 11. 검증 데이터로 앙상블 가중치 탐색
# ============================================================
val_lstm = scaler_y.inverse_transform(lstm_model.predict(X_val, verbose=0)).ravel()
val_conv = scaler_y.inverse_transform(conv_model.predict(X_val, verbose=0)).ravel()

weights = np.linspace(0.0, 1.0, 101)
ensemble_val_mae = []

for w in weights:
    ensemble_val = w * val_lstm + (1.0 - w) * val_conv
    ensemble_val_mae.append(mean_absolute_error(y_val_actual, ensemble_val))

best_idx = int(np.argmin(ensemble_val_mae))
best_w = float(weights[best_idx])
best_conv_w = 1.0 - best_w

print('\n===== ENSEMBLE WEIGHT =====')
print(f'LSTM weight  : {best_w:.2f}')
print(f'Conv1D weight: {best_conv_w:.2f}')
print(f'Validation Ensemble MAE: {ensemble_val_mae[best_idx]:,.2f}원')

# ============================================================
# 12. 테스트 데이터 예측 + 평가
# ============================================================
test_lstm = scaler_y.inverse_transform(lstm_model.predict(X_test, verbose=0)).ravel()
test_conv = scaler_y.inverse_transform(conv_model.predict(X_test, verbose=0)).ravel()
test_ensemble = best_w * test_lstm + best_conv_w * test_conv

# Baseline: 현재 종가를 다음 시가로 예측
baseline_test = df['삼성전자_종가'].reindex(data.index).to_numpy()[target_positions[test_mask]]

results = []
for name, pred in [('LSTM', test_lstm), ('Conv1D', test_conv),
                   ('Ensemble', test_ensemble), ('Baseline', baseline_test)]:
    mae  = mean_absolute_error(y_test_actual, pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, pred))
    results.append({'Model': name, 'MAE': mae, 'RMSE': rmse})

results_df = pd.DataFrame(results)
print('\n===== TEST PERFORMANCE =====')
print(results_df.round(2))

# ============================================================
# 13. 시각화
# ============================================================
plt.figure(figsize=(15, 6))
plt.plot(y_test_actual, label='Actual Open')
plt.plot(test_lstm, label='LSTM')
plt.plot(test_conv, label='Conv1D')
plt.plot(test_ensemble, label='Ensemble')
plt.plot(baseline_test, label='Baseline', alpha=0.7)
plt.title('Samsung Next-Day Open: Model Comparison')
plt.xlabel('Test Sample')
plt.ylabel('Price (KRW)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(history_lstm.history['val_loss'], label='LSTM Val Loss')
plt.plot(history_conv.history['val_loss'], label='Conv1D Val Loss')
plt.title('Validation Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# 14. 최신 데이터로 다음 거래일 예측
# ============================================================
latest_data = df[feature_cols].ffill().tail(TIMESTEPS)

if len(latest_data) != TIMESTEPS:
    raise ValueError('최신 데이터 30개가 부족합니다.')

latest_scaled = scaler_X.transform(latest_data.values).astype(np.float32)
latest_scaled = latest_scaled.reshape(1, TIMESTEPS, len(feature_cols))

next_lstm = scaler_y.inverse_transform(lstm_model.predict(latest_scaled, verbose=0))[0, 0]
next_conv = scaler_y.inverse_transform(conv_model.predict(latest_scaled, verbose=0))[0, 0]
next_ensemble = best_w * next_lstm + best_conv_w * next_conv

print('\n===== NEXT TRADING DAY PREDICTION =====')
print('최종 관측일:', df.index[-1].strftime('%Y-%m-%d'))
print(f'LSTM 예측 시가    : {next_lstm:,.0f}원')
print(f'Conv1D 예측 시가  : {next_conv:,.0f}원')
print(f'Ensemble 예측 시가: {next_ensemble:,.0f}원')

# ============================================================
# 15. 모델 및 결과 저장
# ============================================================
save_path = path + 'CRAZY_AI_SAMSUNG_V2/'
os.makedirs(save_path, exist_ok=True)

# 모델 저장
lstm_model.save(save_path + 'samsung_lstm_model.keras')
conv_model.save(save_path + 'samsung_conv1d_model.keras')

# 가중치만 별도 저장
lstm_model.save_weights(save_path + 'samsung_lstm_weights.weights.h5')
conv_model.save_weights(save_path + 'samsung_conv1d_weights.weights.h5')

# Scaler 저장
joblib.dump(scaler_X, save_path + 'scaler_X.pkl')
joblib.dump(scaler_y, save_path + 'scaler_y.pkl')

# 설정 저장
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

# 성능 지표 / 히스토리 / 예측 결과 저장
results_df.to_csv(save_path + 'test_metrics.csv', index=False, encoding='utf-8-sig')

pd.DataFrame({
    'LSTM_Weight': weights,
    'Conv1D_Weight': 1.0 - weights,
    'Validation_MAE': ensemble_val_mae
}).to_csv(save_path + 'ensemble_weight_search.csv', index=False, encoding='utf-8-sig')

pd.DataFrame(history_lstm.history).to_csv(save_path + 'lstm_history.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(history_conv.history).to_csv(save_path + 'conv1d_history.csv', index=False, encoding='utf-8-sig')

pd.DataFrame({
    'Target_Date': test_target_dates,
    'Actual_Open': y_test_actual,
    'LSTM_Prediction': test_lstm,
    'Conv1D_Prediction': test_conv,
    'Ensemble_Prediction': test_ensemble,
    'Baseline_Prediction': baseline_test
}).to_csv(save_path + 'test_predictions.csv', index=False, encoding='utf-8-sig')

pd.DataFrame([{
    'Last_Observation_Date': df.index[-1],
    'LSTM_Prediction': next_lstm,
    'Conv1D_Prediction': next_conv,
    'Ensemble_Prediction': next_ensemble,
    'LSTM_Weight': best_w,
    'Conv1D_Weight': best_conv_w
}]).to_csv(save_path + 'next_prediction.csv', index=False, encoding='utf-8-sig')

print('\n모델 및 결과 저장 완료:', save_path)

# ============================================================
# 16. 저장한 모델 불러오기 및 검증
# ============================================================
loaded_lstm = load_model(save_path + 'samsung_lstm_model.keras')
loaded_conv = load_model(save_path + 'samsung_conv1d_model.keras')

print('\n===== MODEL RESTORE CHECK =====')
print('LSTM 모델 복원:', np.allclose(
    lstm_model.predict(X_test[:5], verbose=0),
    loaded_lstm.predict(X_test[:5], verbose=0),
    rtol=1e-5, atol=1e-6
))
print('Conv1D 모델 복원:', np.allclose(
    conv_model.predict(X_test[:5], verbose=0),
    loaded_conv.predict(X_test[:5], verbose=0),
    rtol=1e-5, atol=1e-6
))

print('\nCRAZY AI SAMSUNG STOCK PREDICTION V2.0 완료')