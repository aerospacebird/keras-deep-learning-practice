import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import time

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPool2D, BatchNormalization,
    GlobalAveragePooling2D, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
# ============================================================
# 1. 경로 및 폴더 확인
# ============================================================
path = './_data/image/man_woman/'
IMG_SIZE = (100, 100)

print("===== 폴더 구조 확인 =====")
print("하위 항목:", os.listdir(path))

for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)
    if os.path.isdir(folder_path):
        img_count = len([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))
        ])
        print(f"  {folder}: {img_count}장")

# ============================================================
# 2. 데이터 로드 (전체 한 번에)
# ============================================================
datagen_no_aug = ImageDataGenerator(rescale=1./255)

# 먼저 샘플 수만 확인
temp_gen = datagen_no_aug.flow_from_directory(
    path,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False
)

total_samples = temp_gen.samples
print(f"\n총 이미지 수: {total_samples}")
print("클래스 매핑:", temp_gen.class_indices)

if total_samples == 0:
    raise ValueError("이미지가 하나도 없습니다. 경로와 폴더를 확인하세요.")

# 전체 데이터를 한 번에 로드
generator = datagen_no_aug.flow_from_directory(
    path,
    target_size=IMG_SIZE,
    batch_size=total_samples,   # 전체 개수만큼
    class_mode='binary',
    color_mode='rgb',
    shuffle=False,
    seed=42
)

x_all, y_all = next(generator)
print("x_all shape:", x_all.shape)
print("클래스 분포:", np.unique(y_all, return_counts=True))

# ============================================================
# 3. 남자 / 여자 분리
# ============================================================
class_indices = generator.class_indices   # 예: {'man': 0, 'woman': 1}

man_class = None
woman_class = None

for name, idx in class_indices.items():
    name_lower = name.lower()
    if ('man' in name_lower or 'male' in name_lower or '남' in name_lower) and \
       ('woman' not in name_lower and 'female' not in name_lower and '여' not in name_lower):
        man_class = idx
    elif 'woman' in name_lower or 'female' in name_lower or '여' in name_lower:
        woman_class = idx

print(f"남자 클래스 번호: {man_class}, 여자 클래스 번호: {woman_class}")

if man_class is None or woman_class is None:
    raise ValueError(
        f"man / woman 폴더를 제대로 찾지 못했습니다.\n"
        f"현재 감지된 클래스: {class_indices}\n"
        f"폴더 이름을 'man', 'woman' 으로 맞춰주세요."
    )

man_idx   = np.where(y_all == man_class)[0]
woman_idx = np.where(y_all == woman_class)[0]

x_man   = x_all[man_idx]
y_man   = y_all[man_idx]
x_woman = x_all[woman_idx]
y_woman = y_all[woman_idx]

print(f"\n남자 데이터: {len(x_man)}장")
print(f"여자 데이터: {len(x_woman)}장")

if len(x_woman) == 0:
    raise ValueError(
        "여자 이미지가 0장입니다.\n"
        "1) woman 폴더가 있는지 확인\n"
        "2) 폴더 안에 .jpg / .png 파일이 있는지 확인\n"
        "3) 폴더 이름이 man / woman 인지 확인"
    )

# ============================================================
# 4. 여자 데이터만 증강 (남자 수와 동일하게)
# ============================================================
aug_datagen = ImageDataGenerator(
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=15,
    zoom_range=0.15,
    shear_range=0.1,
    fill_mode='nearest'
)

target_count = len(x_man)
need_augment = max(0, target_count - len(x_woman))

print(f"추가로 필요한 여자 증강 데이터: {need_augment}장")

if need_augment > 0:
    rand_idx = np.random.randint(0, len(x_woman), size=need_augment)
    x_to_aug = x_woman[rand_idx].copy()
    y_to_aug = np.full(need_augment, woman_class, dtype=y_woman.dtype)

    # 증강 수행
    aug_flow = aug_datagen.flow(
        x_to_aug,
        y_to_aug,
        batch_size=need_augment,
        shuffle=False
    )
    x_woman_aug = next(aug_flow)[0]

    x_woman_final = np.concatenate([x_woman, x_woman_aug], axis=0)
    y_woman_final = np.concatenate([y_woman, y_to_aug], axis=0)
else:
    x_woman_final = x_woman.copy()
    y_woman_final = y_woman.copy()

print(f"증강 후 여자 데이터: {len(x_woman_final)}장") #증강 후 여자 데이터: 17678장

# ============================================================
# 5. 최종 학습 데이터 합치기 + 셔플
# ============================================================
x_train = np.concatenate([x_man, x_woman_final], axis=0)
y_train = np.concatenate([y_man, y_woman_final], axis=0)

# 레이블을 0(man) / 1(woman)으로 통일
y_train = np.where(y_train == man_class, 0, 1).astype('float32')

# 셔플
shuffle_idx = np.random.permutation(len(x_train))
x_train = x_train[shuffle_idx]
y_train = y_train[shuffle_idx]

print("\n===== 최종 학습 데이터 =====")
print("x_train:", x_train.shape) #x_train: (35356, 100, 100, 3)
print("y_train:", y_train.shape) #y_train: (35356,)
print("클래스 분포 (0=man, 1=woman):", np.unique(y_train, return_counts=True))
#클래스 분포 (0=man, 1=woman): (array([0., 1.], dtype=float32), array([17678, 17678], dtype=int64))
# ============================================================
# 6. Train / Validation / Test 분리
# ============================================================

# 먼저 Train 80% / Test 20% 분리
x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

# Train 80% 중에서 Validation 20% 분리
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

print("\n===== Train / Validation / Test 분리 후 =====")
print("x_train:", x_train.shape, "y_train:", y_train.shape)
print("x_val  :", x_val.shape,   "y_val  :", y_val.shape)
print("x_test :", x_test.shape,  "y_test :", y_test.shape)

print("train 클래스 분포:", np.unique(y_train, return_counts=True))
print("val   클래스 분포:", np.unique(y_val, return_counts=True))
print("test  클래스 분포:", np.unique(y_test, return_counts=True))


# ============================================================
# 모델 구성 (Functional API)
# ============================================================

input1 = Input(shape=(100, 100, 3), name='input1')


# ============================================================
# Conv Block 1
# ============================================================

conv1_output = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1')(input1)
bn1_output = BatchNormalization(name='bn1')(conv1_output)
conv2_output = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv2')(bn1_output)
bn2_output = BatchNormalization(name='bn2')(conv2_output)
pool1_output = MaxPool2D((2, 2), name='pool1')(bn2_output)
dropout1_output = Dropout(0.15, name='dropout1')(pool1_output)


# ============================================================
# Conv Block 2
# ============================================================

conv3_output = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv3')(dropout1_output)
bn3_output = BatchNormalization(name='bn3')(conv3_output)
conv4_output = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv4')(bn3_output)
bn4_output = BatchNormalization(name='bn4')(conv4_output)
pool2_output = MaxPool2D((2, 2), name='pool2')(bn4_output)
dropout2_output = Dropout(0.20, name='dropout2')(pool2_output)


# ============================================================
# Conv Block 3
# ============================================================

conv5_output = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv5')(dropout2_output)
bn5_output = BatchNormalization(name='bn5')(conv5_output)
conv6_output = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv6')(bn5_output)
bn6_output = BatchNormalization(name='bn6')(conv6_output)
pool3_output = MaxPool2D((2, 2), name='pool3')(bn6_output)
dropout3_output = Dropout(0.25, name='dropout3')(pool3_output)


# ============================================================
# Conv Block 4
# ============================================================

conv7_output = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv7')(dropout3_output)
bn7_output = BatchNormalization(name='bn7')(conv7_output)
pool4_output = MaxPool2D((2, 2), name='pool4')(bn7_output)
dropout4_output = Dropout(0.30, name='dropout4')(pool4_output)


# ============================================================
# Global Average Pooling
# ============================================================

gap_output = GlobalAveragePooling2D(name='GAP')(dropout4_output)


# ============================================================
# Fully Connected Layer
# ============================================================

dense1_output = Dense(128, activation='relu', name='dense1')(gap_output)
bn_dense_output = BatchNormalization(name='bn_dense')(dense1_output)
dropout5_output = Dropout(0.40, name='dropout5')(bn_dense_output)


# ============================================================
# Output Layer
# ============================================================

output1 = Dense(2, activation='softmax', name='output1')(dropout5_output)


# ============================================================
# Functional API Model
# ============================================================

model = Model(inputs=input1, outputs=output1)


# ============================================================
# Model Summary
# ============================================================

model.summary()
# ============================================================
# 모델 구성 (Functional API)
# ============================================================

# input1 = Input(shape=(100, 100, 3), name='input1')

# # Conv Block 1
# x = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1')(input1)
# x = BatchNormalization(name='bn1')(x)
# x = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv2')(x)
# x = BatchNormalization(name='bn2')(x)
# x = MaxPool2D((2, 2), name='pool1')(x)
# x = Dropout(0.15, name='dropout1')(x)

# # Conv Block 2
# x = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv3')(x)
# x = BatchNormalization(name='bn3')(x)
# x = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv4')(x)
# x = BatchNormalization(name='bn4')(x)
# x = MaxPool2D((2, 2), name='pool2')(x)
# x = Dropout(0.20, name='dropout2')(x)

# # Conv Block 3
# x = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv5')(x)
# x = BatchNormalization(name='bn5')(x)
# x = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv6')(x)
# x = BatchNormalization(name='bn6')(x)
# x = MaxPool2D((2, 2), name='pool3')(x)
# x = Dropout(0.25, name='dropout3')(x)

# # Conv Block 4
# x = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv7')(x)
# x = BatchNormalization(name='bn7')(x)
# x = MaxPool2D((2, 2), name='pool4')(x)
# x = Dropout(0.30, name='dropout4')(x)

# # Global Average Pooling
# x = GlobalAveragePooling2D(name='GAP')(x)

# # Fully Connected
# x = Dense(128, activation='relu', name='dense1')(x)
# x = BatchNormalization(name='bn_dense')(x)
# x = Dropout(0.40, name='dropout5')(x)

# # ★ 출력층 (10클래스 + softmax)
# output1 = Dense(2, activation='softmax', name='output1')(x)

# model = Model(inputs=input1, outputs=output1)

# model.summary()


# ============================================================
# 컴파일
# ============================================================
model.compile(
    loss='sparse_categorical_crossentropy',   # 정수 라벨용
    optimizer='adam',
    metrics=['accuracy']
)


# ============================================================
# 훈련
# ============================================================
start_time = time.time()

history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=100,
    verbose=1,
    validation_data=(x_val, y_val)
)
# history = model.fit(
#     x_train, y_train,
#     epochs=50,                    # 100은 너무 김 → 50으로 권장 (필요시 늘리세요)
#     batch_size=100,
#     verbose=1,
#     validation_split=0.2
# )

end_time = time.time()


# ============================================================
# 평가
# ============================================================
print("================================ model.evaluate ================================")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print('loss :', round(test_loss, 4))
print('acc  :', round(test_acc, 4))


# ============================================================
# 예측
# ============================================================
y_predict_prob = model.predict(x_test, verbose=0)
y_predict = np.argmax(y_predict_prob, axis=1)          # ★ 2클래스용

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score :', round(acc_score, 4))
print('걸린시간 :', round(end_time - start_time, 2), '초')

##############################################  RESULT  #########################################################
# 221/221 [==============================] - 2s 7ms/step - loss: 0.1888 - accuracy: 0.9548
# loss : 0.1888
# acc  : 0.9548
# 
# accuracy_score : 0.9548
# 걸린시간 : 991.54 초

#######################################  새로운 유형을 적용하여 코드를 작성하다.###########################################################################
# import os
# import numpy as np
# import time
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import (
#     Input, Conv2D, MaxPool2D, BatchNormalization,
#     GlobalAveragePooling2D, Dense, Dropout
# )
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# # ============================================================
# # 0. 설정
# # ============================================================
# path = './_data/image/man_woman/'
# IMG_SIZE = (100, 100)
# SEED = 42
# np.random.seed(SEED)

# # ============================================================
# # 1. 폴더 구조 확인 + 클래스 자동 감지
# # ============================================================
# print("=" * 60)
# print("1. 폴더 구조 확인")
# print("=" * 60)

# if not os.path.exists(path):
#     raise FileNotFoundError(f"경로가 존재하지 않습니다: {path}")

# print("하위 항목:", os.listdir(path))

# for folder in os.listdir(path):
#     folder_path = os.path.join(path, folder)
#     if os.path.isdir(folder_path):
#         img_count = len([
#             f for f in os.listdir(folder_path)
#             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))
#         ])
#         print(f"  {folder}: {img_count}장")

# # ============================================================
# # 2. 데이터 로드 (전체 한 번에 로드 방식 - 현재 데이터 크기에 적합)
# # ============================================================
# print("\n" + "=" * 60)
# print("2. 데이터 로드 (전체 메모리 로드)")
# print("=" * 60)

# datagen_no_aug = ImageDataGenerator(rescale=1./255)

# # 샘플 수 확인용
# temp_gen = datagen_no_aug.flow_from_directory(
#     path,
#     target_size=IMG_SIZE,
#     batch_size=1,
#     class_mode='binary',
#     color_mode='rgb',
#     shuffle=False
# )

# total_samples = temp_gen.samples
# print(f"총 이미지 수: {total_samples}")
# print("클래스 매핑:", temp_gen.class_indices)

# if total_samples == 0:
#     raise ValueError("이미지가 하나도 없습니다. 경로와 폴더를 확인하세요.")

# # 전체 로드
# generator = datagen_no_aug.flow_from_directory(
#     path,
#     target_size=IMG_SIZE,
#     batch_size=total_samples,
#     class_mode='binary',
#     color_mode='rgb',
#     shuffle=False,
#     seed=SEED
# )

# x_all, y_all = next(generator)
# print("x_all shape:", x_all.shape)
# print("클래스 분포:", np.unique(y_all, return_counts=True))

# # ============================================================
# # 3. 남자 / 여자 클래스 자동 감지 + 분리
# # ============================================================
# print("\n" + "=" * 60)
# print("3. 남자 / 여자 분리")
# print("=" * 60)

# class_indices = generator.class_indices  # 예: {'man': 0, 'woman': 1}

# man_class = None
# woman_class = None

# for name, idx in class_indices.items():
#     name_lower = name.lower()
#     # 남자 감지 (woman/female/여가 포함되지 않은 경우만)
#     if any(k in name_lower for k in ['man', 'male', '남']) and \
#        not any(k in name_lower for k in ['woman', 'female', '여']):
#         man_class = idx
#     # 여자 감지
#     elif any(k in name_lower for k in ['woman', 'female', '여']):
#         woman_class = idx

# print(f"남자 클래스 번호: {man_class}, 여자 클래스 번호: {woman_class}")

# if man_class is None or woman_class is None:
#     raise ValueError(
#         f"man / woman 폴더를 제대로 찾지 못했습니다.\n"
#         f"현재 감지된 클래스: {class_indices}\n"
#         f"폴더 이름을 'man', 'woman' (또는 male/female)으로 맞춰주세요."
#     )

# man_idx   = np.where(y_all == man_class)[0]
# woman_idx = np.where(y_all == woman_class)[0]

# x_man   = x_all[man_idx]
# y_man   = y_all[man_idx]
# x_woman = x_all[woman_idx]
# y_woman = y_all[woman_idx]

# print(f"남자 데이터: {len(x_man)}장")
# print(f"여자 데이터: {len(x_woman)}장")

# if len(x_woman) == 0:
#     raise ValueError(
#         "여자 이미지가 0장입니다.\n"
#         "1) woman 폴더가 있는지 확인\n"
#         "2) 폴더 안에 .jpg / .png 파일이 있는지 확인\n"
#         "3) 폴더 이름이 man / woman 인지 확인"
#     )

# # ============================================================
# # 4. 여자 데이터만 증강 (남자 수와 동일하게) - 다양성 강화 버전
# # ============================================================
# print("\n" + "=" * 60)
# print("4. 여자 데이터 증강 (부족한 양만 생성)")
# print("=" * 60)

# # 증강 강도 조절 가능 (너무 세면 왜곡, 너무 약하면 효과 적음)
# aug_datagen = ImageDataGenerator(
#     horizontal_flip=True,
#     width_shift_range=0.12,
#     height_shift_range=0.12,
#     rotation_range=20,
#     zoom_range=0.18,
#     shear_range=0.12,
#     brightness_range=[0.85, 1.15],   # 밝기 변화 추가
#     fill_mode='nearest'
# )

# target_count = len(x_man)
# need_augment = max(0, target_count - len(x_woman))

# print(f"추가로 필요한 여자 증강 데이터: {need_augment}장")

# if need_augment > 0:
#     # 원본 여자 이미지에서 랜덤으로 뽑아서 증강 (중복 허용)
#     rand_idx = np.random.randint(0, len(x_woman), size=need_augment)
#     x_to_aug = x_woman[rand_idx].copy()
#     y_to_aug = np.full(need_augment, woman_class, dtype=y_woman.dtype)

#     # 한 번에 증강 (batch_size = need_augment)
#     aug_flow = aug_datagen.flow(
#         x_to_aug,
#         y_to_aug,
#         batch_size=need_augment,
#         shuffle=False
#     )
#     x_woman_aug, _ = next(aug_flow)

#     x_woman_final = np.concatenate([x_woman, x_woman_aug], axis=0)
#     y_woman_final = np.concatenate([y_woman, y_to_aug], axis=0)
# else:
#     x_woman_final = x_woman.copy()
#     y_woman_final = y_woman.copy()

# print(f"증강 후 여자 데이터: {len(x_woman_final)}장")

# # ============================================================
# # 5. 최종 학습 데이터 합치기 + 레이블 통일 + 셔플
# # ============================================================
# print("\n" + "=" * 60)
# print("5. 최종 데이터 합치기 + 셔플")
# print("=" * 60)

# x_full = np.concatenate([x_man, x_woman_final], axis=0)
# y_full = np.concatenate([y_man, y_woman_final], axis=0)

# # 레이블을 0(man) / 1(woman)으로 통일
# y_full = np.where(y_full == man_class, 0, 1).astype('float32')

# # 셔플
# shuffle_idx = np.random.permutation(len(x_full))
# x_full = x_full[shuffle_idx]
# y_full = y_full[shuffle_idx]

# print("x_full:", x_full.shape)
# print("y_full:", y_full.shape)
# print("클래스 분포 (0=man, 1=woman):", np.unique(y_full, return_counts=True))

# # ============================================================
# # 6. Train / Validation / Test 분리 (3-way split)
# # ============================================================
# print("\n" + "=" * 60)
# print("6. Train / Val / Test 분리")
# print("=" * 60)

# # 방법 1: 한 번에 분리하고 싶을 때 (주석 처리된 버전도 가능)
# # 여기서는 기존 방식 유지하면서 더 명확하게 작성

# # 1단계: Train+Val (80%) / Test (20%)
# x_temp, x_test, y_temp, y_test = train_test_split(
#     x_full, y_full,
#     test_size=0.20,
#     random_state=SEED,
#     stratify=y_full
# )

# # 2단계: Train (64%) / Val (16%)  ← 전체 기준 80% * 0.2 = 16%
# x_train, x_val, y_train, y_val = train_test_split(
#     x_temp, y_temp,
#     test_size=0.20,
#     random_state=SEED,
#     stratify=y_temp
# )

# print("x_train:", x_train.shape, "y_train:", y_train.shape)
# print("x_val  :", x_val.shape,   "y_val  :", y_val.shape)
# print("x_test :", x_test.shape,  "y_test :", y_test.shape)

# print("train 클래스 분포:", np.unique(y_train, return_counts=True))
# print("val   클래스 분포:", np.unique(y_val, return_counts=True))
# print("test  클래스 분포:", np.unique(y_test, return_counts=True))

# # ============================================================
# # (선택) 메모리 절약용 대안 - Generator 방식 (필요시 주석 해제)
# # ============================================================
# # 만약 메모리가 부족하면 아래처럼 flow를 사용하는 방식으로 전환 가능
# # (이 경우 모델 fit 시 x_train, y_train 대신 generator를 넘겨야 함)
# #
# # train_datagen = ImageDataGenerator(
# #     rescale=1./255,
# #     horizontal_flip=True,
# #     width_shift_range=0.1,
# #     height_shift_range=0.1,
# #     rotation_range=15,
# #     zoom_range=0.15
# # )
# # val_datagen = ImageDataGenerator(rescale=1./255)
# #
# # # 이 경우 폴더 구조를 train/man, train/woman, val/man ... 으로 미리 나눠야 함
# # # 또는 전체 로드 후 인덱스 기반으로 커스텀 generator를 만드는 방법도 있음

# print("\n데이터 준비 완료! 이제 모델 구성으로 넘어갑니다.")
# print("=" * 60)

# # ============================================================
# # 모델 구성 (Functional API) - 기존 코드 그대로
# # ============================================================

# input1 = Input(shape=(100, 100, 3), name='input1')

# # Conv Block 1
# conv1_output = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1')(input1)
# bn1_output = BatchNormalization(name='bn1')(conv1_output)
# conv2_output = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv2')(bn1_output)
# bn2_output = BatchNormalization(name='bn2')(conv2_output)
# pool1_output = MaxPool2D((2, 2), name='pool1')(bn2_output)
# dropout1_output = Dropout(0.15, name='dropout1')(pool1_output)

# # Conv Block 2
# conv3_output = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv3')(dropout1_output)
# bn3_output = BatchNormalization(name='bn3')(conv3_output)
# conv4_output = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv4')(bn3_output)
# bn4_output = BatchNormalization(name='bn4')(conv4_output)
# pool2_output = MaxPool2D((2, 2), name='pool2')(bn4_output)
# dropout2_output = Dropout(0.20, name='dropout2')(pool2_output)

# # Conv Block 3
# conv5_output = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv5')(dropout2_output)
# bn5_output = BatchNormalization(name='bn5')(conv5_output)
# conv6_output = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv6')(bn5_output)
# bn6_output = BatchNormalization(name='bn6')(conv6_output)
# pool3_output = MaxPool2D((2, 2), name='pool3')(bn6_output)
# dropout3_output = Dropout(0.25, name='dropout3')(pool3_output)

# # Conv Block 4
# conv7_output = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv7')(dropout3_output)
# bn7_output = BatchNormalization(name='bn7')(conv7_output)
# pool4_output = MaxPool2D((2, 2), name='pool4')(bn7_output)
# dropout4_output = Dropout(0.30, name='dropout4')(pool4_output)

# # Global Average Pooling
# gap_output = GlobalAveragePooling2D(name='GAP')(dropout4_output)

# # Fully Connected Layer
# dense1_output = Dense(128, activation='relu', name='dense1')(gap_output)
# bn_dense_output = BatchNormalization(name='bn_dense')(dense1_output)
# dropout5_output = Dropout(0.40, name='dropout5')(bn_dense_output)

# # Output Layer
# output1 = Dense(2, activation='softmax', name='output1')(dropout5_output)

# model = Model(inputs=input1, outputs=output1)
# model.summary()

# # ============================================================
# # 컴파일
# # ============================================================
# model.compile(
#     loss='sparse_categorical_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy']
# )

# # ============================================================
# # 훈련 (EarlyStopping + ReduceLROnPlateau 추가 추천)
# # ============================================================
# es = EarlyStopping(
#     monitor='val_loss',
#     patience=8,
#     restore_best_weights=True,
#     verbose=1
# )

# rlr = ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.5,
#     patience=4,
#     min_lr=1e-6,
#     verbose=1
# )

# start_time = time.time()

# history = model.fit(
#     x_train, y_train,
#     epochs=50,
#     batch_size=100,
#     verbose=1,
#     validation_data=(x_val, y_val),
#     callbacks=[es, rlr]
# )

# end_time = time.time()

# # ============================================================
# # 평가
# # ============================================================
# print("=" * 60)
# print("model.evaluate")
# print("=" * 60)
# test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
# print('loss :', round(test_loss, 4))
# print('acc  :', round(test_acc, 4))

# # ============================================================
# # 예측
# # ============================================================
# y_predict_prob = model.predict(x_test, verbose=0)
# y_predict = np.argmax(y_predict_prob, axis=1)

# acc_score = accuracy_score(y_test, y_predict)
# print('accuracy_score :', round(acc_score, 4))
# print('걸린시간 :', round(end_time - start_time, 2), '초')