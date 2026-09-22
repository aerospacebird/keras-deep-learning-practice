import os
import optuna
import numpy as np
import time
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPool2D, BatchNormalization,
    GlobalAveragePooling2D, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================
# 0. 설정
# ============================================================
path = './_data/image/man_woman/'
IMG_SIZE = (100, 100)
SEED = 42
np.random.seed(SEED)

# ============================================================
# 1 ~ 6. 데이터 준비 (기존 코드와 동일)
# ============================================================
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

datagen_no_aug = ImageDataGenerator(rescale=1./255)

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

generator = datagen_no_aug.flow_from_directory(
    path,
    target_size=IMG_SIZE,
    batch_size=total_samples,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False,
    seed=SEED
)

x_all, y_all = next(generator)
print("x_all shape:", x_all.shape)
print("클래스 분포:", np.unique(y_all, return_counts=True))

# 남자 / 여자 분리
class_indices = generator.class_indices

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
    raise ValueError("여자 이미지가 0장입니다.")

# 여자 데이터 증강
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

print(f"증강 후 여자 데이터: {len(x_woman_final)}장")

# 최종 데이터 합치기
x_full = np.concatenate([x_man, x_woman_final], axis=0)
y_full = np.concatenate([y_man, y_woman_final], axis=0)
y_full = np.where(y_full == man_class, 0, 1).astype('float32')

shuffle_idx = np.random.permutation(len(x_full))
x_full = x_full[shuffle_idx]
y_full = y_full[shuffle_idx]

print("\n===== 최종 학습 데이터 =====")
print("x_full:", x_full.shape)
print("y_full:", y_full.shape)
print("클래스 분포 (0=man, 1=woman):", np.unique(y_full, return_counts=True))

# Train / Val / Test 분리
x_temp, x_test, y_temp, y_test = train_test_split(
    x_full, y_full,
    test_size=0.20,
    random_state=SEED,
    stratify=y_full
)

x_train, x_val, y_train, y_val = train_test_split(
    x_temp, y_temp,
    test_size=0.20,
    random_state=SEED,
    stratify=y_temp
)

print("\n===== Train / Validation / Test 분리 후 =====")
print("x_train:", x_train.shape, "y_train:", y_train.shape)
print("x_val  :", x_val.shape,   "y_val  :", y_val.shape)
print("x_test :", x_test.shape,  "y_test :", y_test.shape)

# ============================================================
# Optuna용 모델 생성 함수
# ============================================================
def create_model(trial):
    input1 = Input(shape=(100, 100, 3), name='input1')

    # Conv Block 1
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(input1)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2))(x)
    x = Dropout(0.15)(x)

    # Conv Block 2
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2))(x)
    x = Dropout(0.20)(x)

    # Conv Block 3
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Conv Block 4
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2))(x)
    x = Dropout(0.30)(x)

    # GAP + Dense
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.40)(x)

    output1 = Dense(2, activation='softmax')(x)

    model = Model(inputs=input1, outputs=output1)

    # ----- Optimizer 자동 선택 -----
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop", "adamw"])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    if optimizer_name == "adam":
        optimizer = Adam(learning_rate=lr)
    elif optimizer_name == "sgd":
        momentum = trial.suggest_float("momentum", 0.0, 0.99)
        optimizer = SGD(learning_rate=lr, momentum=momentum)
    elif optimizer_name == "rmsprop":
        optimizer = RMSprop(learning_rate=lr)
    else:  # adamw
        optimizer = AdamW(learning_rate=lr)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model


# ============================================================
# Optuna Objective 함수
# ============================================================
def objective(trial):
    model = create_model(trial)

    es = EarlyStopping(
        monitor='val_accuracy',
        patience=6,
        restore_best_weights=True,
        verbose=0
    )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=40,                  # 탐색 단계에서는 너무 길게 하지 않음
        batch_size=100,
        verbose=0,
        callbacks=[es]
    )

    # 검증 정확도를 최대화
    val_acc = max(history.history['val_accuracy'])
    return val_acc


# ============================================================
# Optuna 실행
# ============================================================
print("\n" + "="*60)
print("Optuna 탐색 시작")
print("="*60)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)   # 필요에 따라 30~50으로 늘리세요

print("\n===== Best Trial =====")
print("Best val_accuracy:", study.best_value)
print("Best params:", study.best_params)


# ============================================================
# 최적 파라미터로 최종 모델 재학습
# ============================================================
print("\n" + "="*60)
print("최적 파라미터로 최종 모델 학습")
print("="*60)

best_trial = study.best_trial
final_model = create_model(best_trial)

es = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True,
    verbose=1
)
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)

start_time = time.time()

history = final_model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    batch_size=100,
    verbose=1,
    callbacks=[es, rlr]
)

end_time = time.time()


# ============================================================
# 최종 평가
# ============================================================
print("\n" + "="*60)
print("최종 테스트 평가")
print("="*60)

test_loss, test_acc = final_model.evaluate(x_test, y_test, verbose=1)
print('loss :', round(test_loss, 4))
print('acc  :', round(test_acc, 4))

y_predict_prob = final_model.predict(x_test, verbose=0)
y_predict = np.argmax(y_predict_prob, axis=1)
acc_score = accuracy_score(y_test, y_predict)

print('accuracy_score :', round(acc_score, 4))
print('걸린시간 :', round(end_time - start_time, 2), '초')
print("사용된 Best params:", study.best_params)