
# import numpy as np

# # 이미지 데이터를 폴더에서 읽어오기 위한 클래스
# from keras.preprocessing.image import ImageDataGenerator

# # ============================================================
# # TensorFlow Keras의 함수형 모델 관련 클래스
# # ============================================================

# # Model : 함수형 API를 이용하여 모델 생성
# from tensorflow.keras.models import Model

# # Input : 입력층 생성
# from tensorflow.keras.layers import Input

# # Conv2D : 2차원 합성곱 연산
# # Dense : 완전연결층
# # Dropout : 과적합 방지
# from tensorflow.keras.layers import Conv2D, Dense, Dropout

# # MaxPool2D : 특징 맵의 크기를 줄이는 최대 풀링
# # GlobalAveragePooling2D : 각 채널의 평균값을 계산하여 1차원으로 변환
# from tensorflow.keras.layers import MaxPool2D, GlobalAveragePooling2D

# # 학습에 걸린 시간을 측정하기 위한 모듈
# import time

# # 실제값과 예측값을 비교하여 정확도 계산
# from sklearn.metrics import accuracy_score


# # ============================================================
# # 1. DATA 준비
# # ============================================================

# # ------------------------------------------------------------
# # Train 데이터에 사용할 ImageDataGenerator
# # ------------------------------------------------------------
# # rescale=1./255
# # 0~255 범위의 픽셀값을 0~1 범위로 변환
# #
# # 예:
# # 0   → 0.0
# # 128 → 0.502
# # 255 → 1.0
# # ------------------------------------------------------------

# train_datagen = ImageDataGenerator(
#     rescale=1./255,
# )


# # ------------------------------------------------------------
# # Test 데이터에 사용할 ImageDataGenerator
# # ------------------------------------------------------------

# test_datagen = ImageDataGenerator(
#     rescale=1./255,
# )


# # ------------------------------------------------------------
# # 데이터가 저장되어 있는 경로
# # ------------------------------------------------------------

# path_train = './_data/image/brain/train/'
# path_test = './_data/image/brain/test/'


# # ============================================================
# # Train 데이터 읽기
# # ============================================================

# xy_train = train_datagen.flow_from_directory(

#     # Train 이미지가 저장된 폴더
#     path_train,

#     # 모든 이미지를 150 × 150 크기로 변경
#     target_size=(150, 150),

#     # 한 번에 가져올 이미지 개수
#     batch_size=160,

#     # 이진분류
#     # 클래스가 2개이므로 0 또는 1로 반환
#     class_mode='binary',

#     # 이미지를 흑백으로 읽음
#     # 채널 수 = 1
#     color_mode='grayscale',

#     # 데이터 순서를 섞지 않음
#     shuffle=False,
# )


# # ============================================================
# # Test 데이터 읽기
# # ============================================================

# xy_test = test_datagen.flow_from_directory(

#     # Test 이미지가 저장된 폴더
#     path_test,

#     # 이미지 크기를 150 × 150으로 변경
#     target_size=(150, 150),

#     # Test 이미지 120장을 한 번에 가져옴
#     batch_size=120,

#     # 이진분류
#     class_mode='binary',

#     # 흑백 이미지
#     color_mode='grayscale',

#     # 데이터 순서를 섞지 않음
#     shuffle=False,
# )


# # ============================================================
# # 첫 번째 Batch에서 X와 Y 추출
# # ============================================================

# # xy_train[0]
# # → 첫 번째 batch
# #
# # xy_train[0][0]
# # → X 데이터
# #
# # xy_train[0][1]
# # → Y 데이터

# x_train = xy_train[0][0]
# y_train = xy_train[0][1]


# # Test 데이터도 동일하게 추출

# x_test = xy_test[0][0]
# y_test = xy_test[0][1]


# # ============================================================
# # 데이터 Shape 확인
# # ============================================================

# print(x_train.shape, y_train.shape)

# # 예상 결과
# # (160, 150, 150, 1) (160,)
# #
# # 160 : 이미지 개수
# # 150 : 이미지 높이
# # 150 : 이미지 너비
# # 1   : 흑백 채널


# print(x_test.shape, y_test.shape)

# # 예상 결과
# # (120, 150, 150, 1) (120,)


# # ============================================================
# # 픽셀값의 최대값과 최소값 확인
# # ============================================================

# print(np.max(x_train), np.min(x_train))
# print(np.max(x_test), np.min(x_test))

# # 일반적인 원본 이미지의 픽셀값
# #
# # 최소값 = 0
# # 최대값 = 255
# #
# # 0   → 검정
# # 255 → 흰색


# # ============================================================
# # 2. Scaling
# # ============================================================

# # ------------------------------------------------------------
# # 픽셀값을 0~255 → 0~1 범위로 변환
# # ------------------------------------------------------------

# x_train = x_train / 255.
# x_test = x_test / 255.


# # Scaling 결과 확인

# print(np.max(x_train), np.min(x_train))
# print(np.max(x_test), np.min(x_test))

# # 결과
# #
# # 최대값 = 1.0
# # 최소값 = 0.0


# # ============================================================
# # 3. CNN 함수형 모델 구성
# # ============================================================

# # ------------------------------------------------------------
# # 입력층
# # ------------------------------------------------------------
# #
# # 입력 이미지 Shape
# #
# # 150 × 150 × 1
# #
# # 150 : Height
# # 150 : Width
# # 1   : Grayscale Channel

# # ============================================================
# # 2. CNN 함수형 모델 구성
# # ============================================================

# # 입력층 : 150 × 150 크기의 흑백 이미지
# input1 = Input(shape=(150, 150, 1), name='input1')

# # 첫 번째 합성곱층 : 3×3 필터 32개, 특징 추출
# x = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1')(input1)

# # 첫 번째 MaxPooling : 이미지 크기를 1/2로 축소
# x = MaxPool2D((2, 2), name='pool1')(x)

# # 두 번째 합성곱층 : 3×3 필터 64개, 더 복잡한 특징 추출
# x = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2')(x)

# # 두 번째 MaxPooling : 이미지 크기를 다시 1/2로 축소
# x = MaxPool2D((2, 2), name='pool2')(x)

# # 세 번째 합성곱층 : 3×3 필터 128개, 고수준 특징 추출
# x = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3')(x)

# # 세 번째 MaxPooling : Feature Map 크기 축소
# x = MaxPool2D((2, 2), name='pool3')(x)

# # Global Average Pooling : 각 Feature Map의 평균값을 계산하여 1차원으로 변환
# x = GlobalAveragePooling2D(name='GAP')(x)

# # 완전연결층 : 추출된 특징을 이용하여 분류
# x = Dense(128, activation='relu', name='dense1')(x)

# # Dropout : 30%의 뉴런을 랜덤하게 제거하여 과적합 방지
# x = Dropout(0.3, name='dropout1')(x)

# # 출력층 : 이진분류이므로 뉴런 1개 + Sigmoid 사용
# output1 = Dense(1, activation='softmax', name='output1')(x)

# # 함수형 모델 생성 : input1 → output1
# model = Model(inputs=input1, outputs=output1)

# # 모델 구조 확인
# model.summary()




# # ============================================================
# # 4. 모델 컴파일
# # ============================================================

# # ------------------------------------------------------------
# # binary_crossentropy
# #
# # 이진분류에서 사용하는 Loss Function
# #
# # 출력:
# # sigmoid → 0~1
# #
# # 정답:
# # 0 또는 1
# # ------------------------------------------------------------

# model.compile(
#     loss='binary_crossentropy',

#     # Adam Optimizer
#     optimizer='adam',

#     # 정확도 출력
#     metrics=['acc']
# )


# # ============================================================
# # 5. 모델 훈련
# # ============================================================

# # 학습 시작 시간 저장

# start_time = time.time()


# # ------------------------------------------------------------
# # 모델 학습
# # ------------------------------------------------------------

# hist = model.fit(

#     # Train X
#     x_train,

#     # Train Y
#     y_train,

#     # 전체 데이터를 50번 반복 학습
#     epochs=50,

#     # 한 번에 32개 이미지씩 학습
#     batch_size=32,

#     # 학습 과정 출력
#     verbose=1,

#     # Train 데이터의 20%를 Validation 데이터로 사용
#     validation_split=0.2,
# )


# # 학습 종료 시간 저장

# end_time = time.time()


# # ============================================================
# # 6. 모델 평가
# # ============================================================

# print()
# print(
#     "=============================== model.evaluate ==============================="
# )


# # ------------------------------------------------------------
# # Test 데이터로 모델 평가
# # ------------------------------------------------------------
# #
# # loss[0]
# # → Test Loss
# #
# # loss[1]
# # → Test Accuracy
# # ------------------------------------------------------------

# loss = model.evaluate(
#     x_test,
#     y_test,
#     verbose=1
# )


# # Loss 출력

# print('loss :', loss[0])


# # Accuracy 출력

# print('acc  :', loss[1])


# # ============================================================
# # 7. 예측
# # ============================================================

# # ------------------------------------------------------------
# # Test 데이터에 대한 예측
# # ------------------------------------------------------------

# y_predict = model.predict(
#     x_test
# )


# # ------------------------------------------------------------
# # sigmoid 출력값을 0 또는 1로 변환
# # ------------------------------------------------------------
# #
# # 0.5보다 크면 → 1
# # 0.5보다 작거나 같으면 → 0
# #
# # 예:
# #
# # 0.23 → 0
# # 0.71 → 1
# # ------------------------------------------------------------

# y_predict = (
#     y_predict > 0.5
# ).astype(int).reshape(-1)


# # 실제 정답도 정수형 1차원 배열로 변환

# y_test = y_test.astype(int).reshape(-1)


# # ============================================================
# # 8. Accuracy Score 계산
# # ============================================================

# # sklearn의 accuracy_score를 이용하여
# # 실제값과 예측값의 정확도를 계산

# acc_score = accuracy_score(
#     y_test,
#     y_predict
# )


# # 정확도 출력

# print()
# print(
#     'accuracy_score :',
#     acc_score
# )


# # ============================================================
# # 9. 학습에 걸린 시간
# # ============================================================

import numpy as np

# 이미지 데이터를 폴더에서 읽어오기 위한 클래스
from keras.preprocessing.image import ImageDataGenerator

# ============================================================
# TensorFlow Keras의 함수형 모델 관련 클래스
# ============================================================

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Dense, Dropout
from tensorflow.keras.layers import MaxPool2D, GlobalAveragePooling2D

# ★ 수정/추가 : BatchNormalization 추가
from tensorflow.keras.layers import BatchNormalization

# ★ 수정/추가 : EarlyStopping, ReduceLROnPlateau 추가
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import time

from sklearn.metrics import accuracy_score


# ============================================================
# 1. DATA 준비
# ============================================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
)


test_datagen = ImageDataGenerator(
    rescale=1./255,
)


path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'


# ============================================================
# Train 데이터 읽기
# ============================================================

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(150, 150),

    # ★ 수정 : 전체 데이터를 여러 Batch로 읽기 위해
    # batch_size를 32로 변경
    batch_size=32,

    class_mode='binary',
    color_mode='grayscale',

    # 기존 코드 유지
    shuffle=False,
)


# ============================================================
# Test 데이터 읽기
# ============================================================

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(150, 150),

    # ★ 수정 : Test 데이터도 여러 Batch로 읽음
    batch_size=32,

    class_mode='binary',
    color_mode='grayscale',
    shuffle=False,
)


# ============================================================
# ★ 수정된 부분
# 전체 Train Batch를 모두 가져온다.
# 기존:
#
# x_train = xy_train[0][0]
# y_train = xy_train[0][1]
#
# 위 코드는 첫 번째 Batch만 사용한다.
# ============================================================

x_train_list = []
y_train_list = []

for i in range(len(xy_train)):

    x_batch, y_batch = xy_train[i]

    x_train_list.append(x_batch)
    y_train_list.append(y_batch)


# ★ 수정 : 모든 Batch를 하나의 NumPy 배열로 결합

x_train = np.concatenate(
    x_train_list,
    axis=0
)

y_train = np.concatenate(
    y_train_list,
    axis=0
)


# ============================================================
# ★ 수정된 부분
# 전체 Test Batch를 모두 가져온다.
#
# 기존:
#
# x_test = xy_test[0][0]
# y_test = xy_test[0][1]
#
# 위 코드는 첫 번째 Test Batch만 사용한다.
# ============================================================

x_test_list = []
y_test_list = []

for i in range(len(xy_test)):

    x_batch, y_batch = xy_test[i]

    x_test_list.append(x_batch)
    y_test_list.append(y_batch)


# ★ 수정 : 모든 Test Batch를 하나의 NumPy 배열로 결합

x_test = np.concatenate(
    x_test_list,
    axis=0
)

y_test = np.concatenate(
    y_test_list,
    axis=0
)


# ============================================================
# 2. 데이터 Shape 확인
# ============================================================

print()
print('==================== DATA SHAPE ====================')

print(
    'x_train shape :',
    x_train.shape
)

print(
    'y_train shape :',
    y_train.shape
)

print(
    'x_test shape  :',
    x_test.shape
)

print(
    'y_test shape  :',
    y_test.shape
)


# ============================================================
# 3. Scaling 확인
# ============================================================

print()
print('==================== DATA RANGE ====================')

print(
    'x_train max :',
    np.max(x_train)
)

print(
    'x_train min :',
    np.min(x_train)
)

print(
    'x_test max :',
    np.max(x_test)
)

print(
    'x_test min :',
    np.min(x_test)
)


# ============================================================
# ★ 중요 수정
# ============================================================
#
# ImageDataGenerator에서 이미
#
# rescale=1./255
#
# 를 적용했기 때문에
#
# x_train = x_train / 255.
# x_test = x_test / 255.
#
# 를 다시 적용하면 안 된다.
#
# 따라서 기존의 아래 코드는 삭제한다.
#
# x_train = x_train / 255.
# x_test = x_test / 255.
#
# ============================================================


# ============================================================
# 4. CNN 함수형 모델 구성
# ============================================================

input1 = Input(
    shape=(150, 150, 1),
    name='input1'
)


# ============================================================
# 첫 번째 CNN Block
# ============================================================

x = Conv2D(
    32,
    (3, 3),
    padding='same',
    activation='relu',
    name='conv1'
)(input1)


# ★ 추가 : BatchNormalization
x = BatchNormalization(
    name='bn1'
)(x)


x = MaxPool2D(
    (2, 2),
    name='pool1'
)(x)


# ============================================================
# 두 번째 CNN Block
# ============================================================

x = Conv2D(
    64,
    (3, 3),
    padding='same',
    activation='relu',
    name='conv2'
)(x)


# ★ 추가 : BatchNormalization
x = BatchNormalization(
    name='bn2'
)(x)


x = MaxPool2D(
    (2, 2),
    name='pool2'
)(x)


# ============================================================
# 세 번째 CNN Block
# ============================================================

x = Conv2D(
    128,
    (3, 3),
    padding='same',
    activation='relu',
    name='conv3'
)(x)


# ★ 추가 : BatchNormalization
x = BatchNormalization(
    name='bn3'
)(x)


x = MaxPool2D(
    (2, 2),
    name='pool3'
)(x)


# ============================================================
# ★ 추가 : 네 번째 CNN Block
# ============================================================

x = Conv2D(
    256,
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
    name='pool4'
)(x)


# ============================================================
# Global Average Pooling
# ============================================================

x = GlobalAveragePooling2D(
    name='GAP'
)(x)


# ============================================================
# Dense
# ============================================================

x = Dense(
    128,
    activation='relu',
    name='dense1'
)(x)


# ============================================================
# Dropout
# ============================================================

x = Dropout(
    0.3,
    name='dropout1'
)(x)


# ============================================================
# 출력층
# ============================================================

# ★★★★★ 가장 중요한 수정 ★★★★★
#
# 기존:
#
# output1 = Dense(
#     1,
#     activation='softmax',
#     name='output1'
# )(x)
#
# 이진분류에서 Dense(1)에는 softmax가 아니라 sigmoid를
# 사용해야 한다.
#
# 수정:
# Dense(1) + sigmoid
#
# 출력값:
# 0 ~ 1
#
# 0.5 이상 → Class 1
# 0.5 미만 → Class 0
# ============================================================

output1 = Dense(
    1,
    activation='sigmoid',       # ★ 수정 : softmax → sigmoid
    name='output1'
)(x)


# ============================================================
# Model 생성
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
# 5. 모델 컴파일
# ============================================================

model.compile(
    loss='binary_crossentropy',

    optimizer='adam',

    metrics=['acc']
)


# ============================================================
# ★ 추가 : EarlyStopping
# ============================================================

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# ★ 추가 : Learning Rate 자동 조절
# ============================================================

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    factor=0.5,
    patience=10,
    min_lr=1e-7,
    verbose=1
)


# ============================================================
# 6. 모델 훈련
# ============================================================

start_time = time.time()


hist = model.fit(

    x_train,

    y_train,

    # ★ 수정 : 50 → 200
    # 충분한 학습 기회를 제공
    epochs=200,

    # ★ 수정 : 32 유지
    batch_size=32,

    verbose=1,

    # Validation 데이터 20%
    validation_split=0.2,

    # ★ 추가 : 학습 데이터 Shuffle
    shuffle=True,

    # ★ 추가 : Callback
    callbacks=[
        es,
        rlr
    ]
)


end_time = time.time()


# ============================================================
# 7. 학습 시간
# ============================================================

print()

print(
    'Training Time :',
    round(
        end_time - start_time,
        2
    ),
    'seconds'
)


# ============================================================
# 8. 모델 평가
# ============================================================

print()

print(
    '=============================== model.evaluate ==============================='
)


loss = model.evaluate(
    x_test,
    y_test,
    verbose=1
)


print()

print(
    'loss :',
    round(
        loss[0],
        4
    )
)


print(
    'acc :',
    round(
        loss[1],
        4
    )
)


# ============================================================
# 9. 예측
# ============================================================

y_predict_prob = model.predict(
    x_test,
    verbose=0
)


# ============================================================
# 10. 0.5 Threshold
# ============================================================

# ★ 수정 : 이진분류 기준을 명확하게 지정
#
# 0.5 이상 → 1
# 0.5 미만 → 0
#
# np.where()를 사용하여 명확하게 구현

y_predict = np.where(
    y_predict_prob >= 0.5,
    1,
    0
).astype(int).reshape(-1)


# 실제값도 정수형 1차원 배열로 변환

y_test = y_test.astype(
    int
).reshape(-1)


# ============================================================
# 11. Accuracy Score
# ============================================================

acc_score = accuracy_score(
    y_test,
    y_predict
)


print()

print(
    'accuracy_score :',
    round(
        acc_score,
        4
    )
)


# ============================================================
# 12. 실제값 / 예측값 확인
# ============================================================

print()

print(
    '==================== PREDICTION ===================='
)

print(
    'Actual  :',
    y_test
)

print(
    'Predict :',
    y_predict
)


# ============================================================
# 13. 예측 확률 확인
# ============================================================

print()

print(
    'Prediction Probability :'
)

print(
    np.round(
        y_predict_prob.reshape(-1),
        4
    )
)