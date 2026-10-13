
import numpy as np
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Flatten,
    BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================
# 1. 데이터 준비
# ============================================================
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("x_train:", x_train.shape, "y_train:", y_train.shape)  # (50000, 32, 32, 3)
print("x_test :", x_test.shape,  "y_test :", y_test.shape)   # (10000, 32, 32, 3)

print("원본 max/min:", np.max(x_train), np.min(x_train))  # 255 0

# VGG16 전처리 (권장)
x_train = preprocess_input(x_train.astype('float32'))
x_test  = preprocess_input(x_test.astype('float32'))

print("전처리 후 max/min:", np.max(x_train), np.min(x_train))

# One-Hot Encoding
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test  = ohe.transform(y_test.reshape(-1, 1))

print("y_train:", y_train.shape, "y_test:", y_test.shape)  # (50000, 10), (10000, 10)


# ============================================================
# 2. VGG16 전이학습 모델 구성
# ============================================================
base_model = VGG16(
    weights='imagenet',          # ImageNet 사전학습 가중치
    include_top=False,           # 분류기 제거
    input_shape=(32, 32, 3)      # CIFAR-10 크기에 맞춤
)

# 1단계: 전체 동결 (Feature Extractor)
base_model.trainable = False

model = Sequential([
    base_model,
    
    GlobalAveragePooling2D(),    # Flatten보다 파라미터 적고 효과적
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(10, activation='softmax')   # CIFAR-10 → 10 클래스
])

model.summary()


# ============================================================
# 3. 컴파일 & 훈련 (1단계: Feature Extraction)
# ============================================================
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['acc']
)

es = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

print("\n===== 1단계: Feature Extraction (VGG16 동결) =====")
start_time = time.time()

hist = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=128,
    validation_split=0.2,
    callbacks=[es, rlr],
    verbose=1
)

end_time = time.time()
print(f"1단계 걸린 시간: {round(end_time - start_time, 2)}초")


# ============================================================
# 4. Fine-tuning (2단계: 상위 일부 층만 학습)
# ============================================================
print("\n===== 2단계: Fine-tuning (상위 층 일부 학습) =====")

base_model.trainable = True

# 앞쪽 층은 계속 동결, 뒤쪽 일부만 학습
for layer in base_model.layers[:-8]:   # 마지막 8개 층만 학습
    layer.trainable = False

# Fine-tuning은 학습률을 매우 작게
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=1e-5),
    metrics=['acc']
)

start_time2 = time.time()

hist2 = model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=128,
    validation_split=0.2,
    callbacks=[es, rlr],
    verbose=1
)

end_time2 = time.time()
print(f"2단계 걸린 시간: {round(end_time2 - start_time2, 2)}초")
print(f"총 걸린 시간   : {round(end_time2 - start_time, 2)}초")


# ============================================================
# 5. 평가 & 예측
# ============================================================
print("\n=============================== 평가 ===============================")
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss:", loss[0])
print("acc :", loss[1])

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_true, y_pred)
print("accuracy_score:", acc_score)
#######################################################################################
# =============================== 평가 ===============================
# 313/313 [==============================] - 2s 4ms/step - loss: 0.6390 - acc: 0.7912
# loss: 0.6389783620834351
# acc : 0.7911999821662903
# 313/313 [==============================] - 1s 4ms/step
# accuracy_score: 0.7912
#########################################################################################