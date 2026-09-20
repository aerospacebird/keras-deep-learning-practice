

import numpy as np
import time

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Dense, Dropout
from tensorflow.keras.layers import MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import accuracy_score


# ============================================================
# 1. DATA 준비
# ============================================================

train_datagen = ImageDataGenerator(
    rescale=1./255,

    horizontal_flip=True,
    vertical_flip=True,

    width_shift_range=0.1,
    height_shift_range=0.1,

    rotation_range=5,

    zoom_range=0.2,

    shear_range=0.2,

    fill_mode='nearest'
)


# ------------------------------------------------------------
# Test 데이터에는 Augmentation을 적용하지 않는다.
# ------------------------------------------------------------

test_datagen = ImageDataGenerator(
    rescale=1./255
)


# ============================================================
# 데이터 경로
# ============================================================

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'


# ============================================================
# Train Generator
# ============================================================

xy_train = train_datagen.flow_from_directory(
    path_train,

    target_size=(150, 150),

    batch_size=160,

    class_mode='binary',

    color_mode='grayscale',

    shuffle=True
)


# ============================================================
# Test Generator
# ============================================================

xy_test = test_datagen.flow_from_directory(
    path_test,

    target_size=(150, 150),

    batch_size=120,

    class_mode='binary',

    color_mode='grayscale',

    shuffle=False
)


# ============================================================
# Generator에서 데이터 추출
# ============================================================

x_train = xy_train[0][0]
y_train = xy_train[0][1]

x_test = xy_test[0][0]
y_test = xy_test[0][1]


# ============================================================
# Shape 확인
# ============================================================

print()
print("x_train :", x_train.shape)
print("y_train :", y_train.shape)

print("x_test  :", x_test.shape)
print("y_test  :", y_test.shape)


# ============================================================
# Scaling 확인
# ============================================================

print()
print("Train MAX :", round(np.max(x_train), 4))
print("Train MIN :", round(np.min(x_train), 4))

print("Test MAX  :", round(np.max(x_test), 4))
print("Test MIN  :", round(np.min(x_test), 4))


# ============================================================
# 중요
# ============================================================
# ImageDataGenerator에서 이미 rescale=1./255를 적용했으므로
# 아래 코드는 사용하지 않는다.
#
# x_train = x_train / 255.
# x_test = x_test / 255.


# ============================================================
# Class 확인
# ============================================================

print()
print("class_indices :", xy_train.class_indices)


# ============================================================
# 2. CNN 함수형 모델 구성 - 성능 개선
# ============================================================

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPool2D,
    BatchNormalization, GlobalAveragePooling2D,
    Dense, Dropout
)


# ------------------------------------------------------------
# Input Layer
# ------------------------------------------------------------

input1 = Input(
    shape=(150, 150, 1),
    name='input1'
)


# ------------------------------------------------------------
# Conv Block 1
# ------------------------------------------------------------

x = Conv2D(
    32,
    (3, 3),
    padding='same',
    activation='relu',
    name='conv1'
)(input1)

x = BatchNormalization(
    name='bn1'
)(x)

x = Conv2D(
    32,
    (3, 3),
    padding='same',
    activation='relu',
    name='conv2'
)(x)

x = BatchNormalization(
    name='bn2'
)(x)

x = MaxPool2D(
    (2, 2),
    name='pool1'
)(x)

x = Dropout(
    0.15,
    name='dropout1'
)(x)


# ------------------------------------------------------------
# Conv Block 2
# ------------------------------------------------------------

x = Conv2D(
    64,
    (3, 3),
    padding='same',
    activation='relu',
    name='conv3'
)(x)

x = BatchNormalization(
    name='bn3'
)(x)

x = Conv2D(
    64,
    (3, 3),
    padding='same',
    activation='relu',
    name='conv4'
)(x)

x = BatchNormalization(
    name='bn4'
)(x)

x = MaxPool2D(
    (2, 2),
    name='pool2'
)(x)

x = Dropout(
    0.20,
    name='dropout2'
)(x)


# ------------------------------------------------------------
# Conv Block 3
# ------------------------------------------------------------

x = Conv2D(
    128,
    (3, 3),
    padding='same',
    activation='relu',
    name='conv5'
)(x)

x = BatchNormalization(
    name='bn5'
)(x)

x = Conv2D(
    128,
    (3, 3),
    padding='same',
    activation='relu',
    name='conv6'
)(x)

x = BatchNormalization(
    name='bn6'
)(x)

x = MaxPool2D(
    (2, 2),
    name='pool3'
)(x)

x = Dropout(
    0.25,
    name='dropout3'
)(x)


# ------------------------------------------------------------
# Conv Block 4
# ------------------------------------------------------------

x = Conv2D(
    256,
    (3, 3),
    padding='same',
    activation='relu',
    name='conv7'
)(x)

x = BatchNormalization(
    name='bn7'
)(x)

x = MaxPool2D(
    (2, 2),
    name='pool4'
)(x)

x = Dropout(
    0.30,
    name='dropout4'
)(x)


# ------------------------------------------------------------
# Global Average Pooling
# ------------------------------------------------------------

x = GlobalAveragePooling2D(
    name='GAP'
)(x)


# ------------------------------------------------------------
# Fully Connected Layer
# ------------------------------------------------------------

x = Dense(
    128,
    activation='relu',
    name='dense1'
)(x)

x = BatchNormalization(
    name='bn_dense'
)(x)

x = Dropout(
    0.40,
    name='dropout5'
)(x)


# ------------------------------------------------------------
# Output Layer
# Binary Classification
# ------------------------------------------------------------

output1 = Dense(
    1,
    activation='sigmoid',
    name='output1'
)(x)


# ============================================================
# Functional Model
# ============================================================

model = Model(
    inputs=input1,
    outputs=output1
)


# ============================================================
# Model Summary
# ============================================================

model.summary()


# ============================================================
# 3. Compile
# ============================================================

optimizer = Adam(
    learning_rate=0.001
)


model.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['acc']
)


# ============================================================
# 4. Callback
# ============================================================

early_stopping = EarlyStopping(
    monitor='val_loss',

    patience=30,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',

    factor=0.5,

    patience=10,

    min_lr=1e-7,

    verbose=1
)


# ============================================================
# 5. Training
# ============================================================

start_time = time.time()


hist = model.fit(
    x_train,
    y_train,

    epochs=1300,

    batch_size=32,

    verbose=1,

    validation_split=0.2,

    shuffle=True,

    callbacks=[
        early_stopping,
        reduce_lr
    ]
)


end_time = time.time()


# ============================================================
# 6. Evaluation
# ============================================================

print()
print("=" * 70)
print("MODEL EVALUATE")
print("=" * 70)


loss = model.evaluate(
    x_test,
    y_test,
    verbose=1
)


print()
print("loss :", round(loss[0], 6))
print("acc  :", round(loss[1], 4))


# ============================================================
# 7. Prediction
# ============================================================

y_predict = model.predict(
    x_test,
    verbose=0
)


# ------------------------------------------------------------
# Sigmoid 확률값 확인
# ------------------------------------------------------------

print()
print("Prediction Probability")
print(
    np.round(
        y_predict[:10].reshape(-1),
        4
    )
)


# ============================================================
# 8. Binary Classification
# ============================================================
#
# 0.5 이상 → 1
# 0.5 미만 → 0
#
# 정확히 0.5인 경우 → 1
# ============================================================

y_predict = np.where(
    y_predict >= 0.5,
    1,
    0
).astype(int).reshape(-1)


# ------------------------------------------------------------
# 실제 정답
# ------------------------------------------------------------

y_test = y_test.astype(int).reshape(-1)


# ============================================================
# 예측 결과 확인
# ============================================================

print()
print("Predicted Class")
print(y_predict[:20])

print()
print("Actual Class")
print(y_test[:20])


# ============================================================
# 9. Accuracy Score
# ============================================================

acc_score = accuracy_score(
    y_test,
    y_predict
)


print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print(
    "accuracy_score :",
    round(acc_score, 4)
)


# ============================================================
# 10. Training Time
# ============================================================

print(
    "걸린시간 :",
    round(end_time - start_time, 2),
    "초"
)


# ============================================================
# 11. Learning Rate
# ============================================================

learning_rate = float(
    model.optimizer.learning_rate.numpy()
)


print(
    "최종 Learning Rate :",
    round(learning_rate, 1000)
)


# ============================================================
# 12. Best Epoch
# ============================================================

best_epoch = np.argmin(
    hist.history['val_loss']
) + 1


best_val_loss = min(
    hist.history['val_loss']
)


best_val_acc = max(
    hist.history['val_acc']
)


print()
print("=" * 70)
print("BEST RESULT")
print("=" * 70)

print(
    "Best Epoch    :",
    best_epoch
)

print(
    "Best val_loss :",
    round(best_val_loss, 6)
)

print(
    "Best val_acc  :",
    round(best_val_acc, 4)
)


#exit()
# print(xy_train)
# #<keras.preprocessing.image.DirectoryIterator object at 0x0000025188F379D0>
# print(xy_test)
# #<keras.preprocessing.image.DirectoryIterator object at 0x0000025188F379D0>

# print(xy_train.next()) # Iterator 첫번째를 보여줘?
# print(xy_train.next()) # 두번째 Iterator 출력해줘?

# # print(xy_train[0])
# # print(xy_train[1])
# # print(xy_train[2])

# #print(xy_train[0][0]  # 첫번째 배치의 X 데이터가 되겠지요.
# #print(xy_train[0][1]  # 첫번째 배치의 Y 데이터가 되겠지요.
# print(xy_train[0][0].shape) # (10, 100, 100, 1)
# print(xy_train[0][1].shape) #(10,)
# exit()
# #print(xy_train)[16][0]) # 여기서 부터 에러, 이유는 160장이다..? 배치는 10개이니까.

# print(type(xy_train)) # <class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0])) # <class 'tuple'>
# print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
# print(type(xy_train[0][1])) # <class 'numpy.ndarray'>

