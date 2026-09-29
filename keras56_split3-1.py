import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ------------------------------------------------------------------
# 1. Data preparation (identical logic, just cleaned)
# ------------------------------------------------------------------
a = np.array(range(1, 101), dtype=np.float32)
size = 6

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : i + size]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)          # (95, 6)

x = bbb[:, :-1]                 # (95, 5)
y = bbb[:, -1]                  # (95,)

# RNN input shape: (samples, timesteps, features)
x = x.reshape(x.shape[0], x.shape[1], 1)   # (95, 5, 1)

print("x.shape :", x.shape, "  y.shape :", y.shape)

# ------------------------------------------------------------------
# 2. Model – modest capacity increase + LayerNorm for stability
# ------------------------------------------------------------------
model = Sequential([
    SimpleRNN(64, return_sequences=True, input_shape=(5, 1)),
    LayerNormalization(),
    SimpleRNN(32),
    LayerNormalization(),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(
    loss='mse',
    optimizer=Adam(learning_rate=1e-3)
)

# ------------------------------------------------------------------
# 3. Training with proper callbacks + validation split
# ------------------------------------------------------------------
es = EarlyStopping(
    monitor='val_loss',
    patience=40,
    restore_best_weights=True,
    verbose=1
)
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=15,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    x, y,
    epochs=400,
    batch_size=8,                 # larger batch → more stable gradients
    validation_split=0.15,
    callbacks=[es, rlr],
    verbose=1
)

# ------------------------------------------------------------------
# 4. Evaluation
# ------------------------------------------------------------------
loss = model.evaluate(x, y, verbose=0)
print(f"\nFinal train loss : {loss:.6f}")

# ------------------------------------------------------------------
# 5. Multi-step prediction for [101, 102, 103, 104, 105, 106]
# ------------------------------------------------------------------
# Start from the last known window: [96, 97, 98, 99, 100]
current = np.array([96, 97, 98, 99, 100], dtype=np.float32).reshape(1, 5, 1)

preds = []
for _ in range(6):                      # predict the next 6 steps
    next_val = model.predict(current, verbose=0)[0, 0]
    preds.append(next_val)
    # slide the window forward
    current = np.concatenate(
        [current[:, 1:, :], next_val.reshape(1, 1, 1)],
        axis=1
    )

print("\nPredicted sequence [101 … 106]:")
print(np.round(preds, 3))

model.summary()