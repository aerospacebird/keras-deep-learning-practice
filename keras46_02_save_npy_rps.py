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

print(np.__version__)

# ============================================================
# 1. ImageDataGenerator 설정
# ============================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.2,
    shear_range=0.7,
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# ============================================================
# 2. 데이터 경로
# ============================================================
path = './_data/image/rps/'

# ============================================================
# 3. flow_from_directory로 데이터 생성 (★ 3클래스)
# ============================================================
xy_train = train_datagen.flow_from_directory(
    path,
    target_size=(100, 100),
    batch_size=32,                    # 10 → 32 권장
    class_mode='categorical',         # ★ binary → categorical
    color_mode='rgb',
    shuffle=True,
    subset='training',
    seed=42
)

xy_test = test_datagen.flow_from_directory(
    path,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',         # ★ binary → categorical
    color_mode='rgb',
    shuffle=False,
    subset='validation',
    seed=42
)

print(xy_train.class_indices)         # {'paper':0, 'rock':1, 'scissors':2} 확인
print(xy_train)
print(xy_test)
print(xy_train[0][0].shape)           # (batch, 100, 100, 3)
print(xy_train[0][1].shape)           # (batch, 3)



###################################################################################################################
x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

############################################    np.save( )                    ########################################
np_path = './_data/hourse-human_npy/'             #
np.save(np_path + 'keras46_02_x_train.npy' , arr = xy_train[0][0])  # arr = x_train 도가능 
np.save(np_path + 'keras46_02_y_train.npy' , arr = xy_train[0][1])  # arr = y_train 도가능
np.save(np_path + 'keras46_02_x_test.npy' , arr = xy_train[0][0])   # arr = x_test 도가능
np.save(np_path + 'keras46_02_y_test.npy' , arr = xy_train[0][1])   # arr = y_test 도가능
######################################################################################################################



# ============================================================
# 2. CNN 함수형 모델 구성
# ============================================================
input1 = Input(shape=(100, 100, 3), name='input1')

# Conv Block 1
x = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1')(input1)
x = BatchNormalization(name='bn1')(x)
x = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv2')(x)
x = BatchNormalization(name='bn2')(x)
x = MaxPool2D((2, 2), name='pool1')(x)
x = Dropout(0.15, name='dropout1')(x)

# Conv Block 2
x = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv3')(x)
x = BatchNormalization(name='bn3')(x)
x = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv4')(x)
x = BatchNormalization(name='bn4')(x)
x = MaxPool2D((2, 2), name='pool2')(x)
x = Dropout(0.20, name='dropout2')(x)

# Conv Block 3
x = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv5')(x)
x = BatchNormalization(name='bn5')(x)
x = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv6')(x)
x = BatchNormalization(name='bn6')(x)
x = MaxPool2D((2, 2), name='pool3')(x)
x = Dropout(0.25, name='dropout3')(x)

# Conv Block 4
x = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv7')(x)
x = BatchNormalization(name='bn7')(x)
x = MaxPool2D((2, 2), name='pool4')(x)
x = Dropout(0.30, name='dropout4')(x)

# GAP + Dense
x = GlobalAveragePooling2D(name='GAP')(x)
x = Dense(128, activation='relu', name='dense1')(x)
x = BatchNormalization(name='bn_dense')(x)
x = Dropout(0.40, name='dropout5')(x)

# ★ 3클래스 softmax
output1 = Dense(3, activation='softmax', name='output1')(x)

model = Model(inputs=input1, outputs=output1)
model.summary()

# ============================================================
# 3. Compile
# ============================================================
optimizer = Adam(learning_rate=0.001)

model.compile(
    loss='categorical_crossentropy',   # ★ binary → categorical
    optimizer=optimizer,
    metrics=['accuracy']
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
    xy_train,
    epochs=300,                        # 1300은 과함 (EarlyStopping 있으니 괜찮지만)
    validation_data=xy_test,
    verbose=1,
    callbacks=[early_stopping, reduce_lr]
)

end_time = time.time()

# ============================================================
# 6. Evaluation
# ============================================================
print()
print("=" * 70)
print("MODEL EVALUATE")
print("=" * 70)

loss = model.evaluate(xy_test, verbose=1)

print()
print("loss :", round(loss[0], 6))
print("acc  :", round(loss[1], 4))

# ============================================================
# 7. Prediction
# ============================================================
y_predict = model.predict(xy_test, verbose=0)

print()
print("Prediction Probability (first 10)")
print(np.round(y_predict[:10], 4))

# ★ argmax로 클래스 예측
y_predict = np.argmax(y_predict, axis=1)

# 실제 정답 추출
y_test = []
for i in range(len(xy_test)):
    y_test.extend(np.argmax(xy_test[i][1], axis=1))
y_test = np.array(y_test)

print()
print("Predicted Class")
print(y_predict[:20])

print()
print("Actual Class")
print(y_test[:20])

# ============================================================
# 8. Accuracy Score
# ============================================================
acc_score = accuracy_score(y_test, y_predict)

print()
print("=" * 70)
print("RESULT")
print("=" * 70)
print("accuracy_score :", round(acc_score, 4))
print("걸린시간 :", round(end_time - start_time, 2), "초")

# ============================================================
# Learning Rate / Best Epoch
# ============================================================
learning_rate = float(model.optimizer.learning_rate.numpy())
print("최종 Learning Rate :", round(learning_rate, 10))

best_epoch = np.argmin(hist.history['val_loss']) + 1
best_val_loss = min(hist.history['val_loss'])
best_val_acc = max(hist.history['val_accuracy'])

print()
print("=" * 70)
print("BEST RESULT")
print("=" * 70)
print("Best Epoch    :", best_epoch)
print("Best val_loss :", round(best_val_loss, 6))
print("Best val_acc  :", round(best_val_acc, 4))
# import numpy as np
# import time

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import (
#     Input, Conv2D, MaxPool2D, BatchNormalization,
#     GlobalAveragePooling2D, Dense, Dropout
# )
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from sklearn.metrics import accuracy_score

# print(np.__version__)

# # ============================================================
# # 1. ImageDataGenerator 설정
# # ============================================================
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     horizontal_flip=True,
#     vertical_flip=True,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     rotation_range=5,
#     zoom_range=0.2,              # 1.2 → 0.2 로 수정 (너무 극단적)
#     shear_range=0.7,
#     fill_mode='nearest',
#     validation_split=0.2
# )

# test_datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2
# )

# # ============================================================
# # 2. 데이터 경로
# # ============================================================
# path = './_data/image/rps/'

# # ============================================================
# # 3. flow_from_directory로 데이터 생성
# # ============================================================
# xy_train = train_datagen.flow_from_directory(
#     path,
#     target_size=(100, 100),
#     batch_size=10,
#     class_mode='binary',
#     color_mode='rgb',
#     shuffle=True,
#     subset='training',
#     seed=42
# )

# xy_test = test_datagen.flow_from_directory(
#     path,
#     target_size=(100, 100),
#     batch_size=10,
#     class_mode='binary',
#     color_mode='rgb',
#     shuffle=False,
#     subset='validation',
#     seed=42
# )

# print(xy_train)
# print(xy_test)
# print(xy_train[0][0].shape)   # (10, 100, 100, 3)
# print(xy_train[0][1].shape)   # (10,)

# # ============================================================
# # Generator에서 데이터 추출
# # ============================================================

# x_train = xy_train[0][0]
# y_train = xy_train[0][1]

# x_test = xy_test[0][0]
# y_test = xy_test[0][1]


# # ============================================================
# # Shape 확인
# # ============================================================

# print()
# print("x_train :", x_train.shape)
# print("y_train :", y_train.shape)

# print("x_test  :", x_test.shape)
# print("y_test  :", y_test.shape)


# # ============================================================
# # Scaling 확인
# # ============================================================

# print()
# print("Train MAX :", round(np.max(x_train), 4))
# print("Train MIN :", round(np.min(x_train), 4))

# print("Test MAX  :", round(np.max(x_test), 4))
# print("Test MIN  :", round(np.min(x_test), 4))

# #exit()
# # ============================================================
# # 2. CNN 함수형 모델 구성
# # ============================================================
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

# # GAP + Dense
# x = GlobalAveragePooling2D(name='GAP')(x)
# x = Dense(128, activation='relu', name='dense1')(x)
# x = BatchNormalization(name='bn_dense')(x)
# x = Dropout(0.40, name='dropout5')(x)

# output1 = Dense(1, activation='softmax', name='output1')(x)  #activation='sigmoid'

# model = Model(inputs=input1, outputs=output1)
# model.summary()

# # ============================================================
# # 3. Compile
# # ============================================================
# optimizer = Adam(learning_rate=0.001)

# model.compile(
#     loss='binary_crossentropy',
#     optimizer=optimizer,
#     metrics=['accuracy']          # 'acc' → 'accuracy' 권장
# )

# # ============================================================
# # 4. Callback
# # ============================================================
# early_stopping = EarlyStopping(
#     monitor='val_loss',
#     patience=30,
#     restore_best_weights=True,
#     verbose=1
# )

# reduce_lr = ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.5,
#     patience=10,
#     min_lr=1e-7,
#     verbose=1
# )

# # ============================================================
# # 5. Training  (Generator 직접 사용)
# # ============================================================
# start_time = time.time()

# hist = model.fit(
#     xy_train,                          # 한 배치가 아니라 Generator 전체 사용
#     epochs=1300,
#     validation_data=xy_test,           # validation_split 대신 validation_data 사용
#     verbose=1,
#     callbacks=[early_stopping, reduce_lr]
# )

# end_time = time.time()

# # ============================================================
# # 6. Evaluation
# # ============================================================
# print()
# print("=" * 70)
# print("MODEL EVALUATE")
# print("=" * 70)

# loss = model.evaluate(xy_test, verbose=1)

# print()
# print("loss :", round(loss[0], 6))
# print("acc  :", round(loss[1], 4))

# # ============================================================
# # 7. Prediction
# # ============================================================
# y_predict = model.predict(xy_test, verbose=0)

# print()
# print("Prediction Probability")
# print(np.round(y_predict[:10].reshape(-1), 4))

# # 0.5 기준 이진화
# y_predict = np.where(y_predict >= 0.5, 1, 0).astype(int).reshape(-1)

# # 실제 정답 추출 (validation set 전체)
# y_test = []
# for i in range(len(xy_test)):
#     y_test.extend(xy_test[i][1])
# y_test = np.array(y_test).astype(int)

# print()
# print("Predicted Class")
# print(y_predict[:20])

# print()
# print("Actual Class")
# print(y_test[:20])

# # ============================================================
# # 8. Accuracy Score
# # ============================================================
# acc_score = accuracy_score(y_test, y_predict)

# print()
# print("=" * 70)
# print("RESULT")
# print("=" * 70)
# print("accuracy_score :", round(acc_score, 4))

# print("걸린시간 :", round(end_time - start_time, 2), "초")

# # ============================================================
# # Learning Rate / Best Epoch
# # ============================================================
# learning_rate = float(model.optimizer.learning_rate.numpy())
# print("최종 Learning Rate :", round(learning_rate, 10))

# best_epoch = np.argmin(hist.history['val_loss']) + 1
# best_val_loss = min(hist.history['val_loss'])
# best_val_acc = max(hist.history['val_accuracy'])

# print()
# print("=" * 70)
# print("BEST RESULT")
# print("=" * 70)
# print("Best Epoch    :", best_epoch)
# print("Best val_loss :", round(best_val_loss, 6))
# print("Best val_acc  :", round(best_val_acc, 4))
# # import numpy as np
# # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# # import numpy as np
# # import time

# # from keras.preprocessing.image import ImageDataGenerator

# # from tensorflow.keras.models import Model
# # from tensorflow.keras.layers import Input
# # from tensorflow.keras.layers import Conv2D, Dense, Dropout
# # from tensorflow.keras.layers import MaxPool2D, GlobalAveragePooling2D
# # from tensorflow.keras.optimizers import Adam
# # from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# # from sklearn.metrics import accuracy_score
# # print(np.__version__)

# # # ============================================================
# # # 1. ImageDataGenerator 설정
# # # ============================================================
# # train_datagen = ImageDataGenerator(
# #     rescale=1./255,              # 0~1 정규화
# #     horizontal_flip=True,        # 좌우 반전
# #     vertical_flip=True,          # 상하 반전
# #     width_shift_range=0.1,       # 가로 이동
# #     height_shift_range=0.1,      # 세로 이동
# #     rotation_range=5,            # 회전
# #     zoom_range=1.2,              # 확대/축소
# #     shear_range=0.7,             # 전단 변환
# #     fill_mode='nearest',
# #     validation_split=0.2         # 20%를 검증용으로 분리
# # )

# # test_datagen = ImageDataGenerator(
# #     rescale=1./255,
# #     validation_split=0.2         # 같은 비율로 맞춰줌
# # )

# # # ============================================================
# # # 2. 데이터 경로
# # # ============================================================
# # path = './_data/image/horse-human/'   # horses / humans 폴더가 들어있는 경로

# # # ============================================================
# # # 3. flow_from_directory로 데이터 생성
# # # ============================================================
# # xy_train = train_datagen.flow_from_directory(
# #     path,
# #     target_size=(100, 100),      # 원하는 크기로 리사이즈
# #     batch_size=10,
# #     class_mode='binary',         # 이진 분류 (horse / human)
# #     color_mode='rgb',            # 컬러 이미지 (grayscale이 아님)
# #     shuffle=True,
# #     subset='training',           # 80% 훈련 데이터
# #     seed=42
# # )

# # xy_test = test_datagen.flow_from_directory(
# #     path,
# #     target_size=(100, 100),
# #     batch_size=10,
# #     class_mode='binary',
# #     color_mode='rgb',
# #     shuffle=False,
# #     subset='validation',         # 20% 검증 데이터
# #     seed=42
# # )

# # # ============================================================
# # # 4. 확인용 출력
# # # ============================================================
# # print(xy_train)
# # print(xy_test)

# # print(xy_train[0][0].shape)   # (10, 100, 100, 3)
# # print(xy_train[0][1].shape)   # (10,)

# # print(type(xy_train))         # <class 'keras...DirectoryIterator'>
# # print(type(xy_train[0]))      # <class 'tuple'>
# # print(type(xy_train[0][0]))   # <class 'numpy.ndarray'>
# # print(type(xy_train[0][1]))   # <class 'numpy.ndarray'>

# # # ============================================================
# # # Generator에서 데이터 추출
# # # ============================================================

# # x_train = xy_train[0][0]
# # y_train = xy_train[0][1]

# # x_test = xy_test[0][0]
# # y_test = xy_test[0][1]


# # # ============================================================
# # # Shape 확인
# # # ============================================================

# # print()
# # print("x_train :", x_train.shape)
# # print("y_train :", y_train.shape)

# # print("x_test  :", x_test.shape)
# # print("y_test  :", y_test.shape)


# # # ============================================================
# # # Scaling 확인
# # # ============================================================

# # print()
# # print("Train MAX :", round(np.max(x_train), 4))
# # print("Train MIN :", round(np.min(x_train), 4))

# # print("Test MAX  :", round(np.max(x_test), 4))
# # print("Test MIN  :", round(np.min(x_test), 4))

# # #exit()
# # # ============================================================
# # # 2. CNN 함수형 모델 구성 - 성능 개선
# # # ============================================================

# # from tensorflow.keras.models import Model
# # from tensorflow.keras.layers import (
# #     Input, Conv2D, MaxPool2D,
# #     BatchNormalization, GlobalAveragePooling2D,
# #     Dense, Dropout
# # )


# # # ------------------------------------------------------------
# # # Input Layer
# # # ------------------------------------------------------------

# # input1 = Input(
# #     shape=(100, 100, 3),
# #     name='input1'
# # )


# # # ------------------------------------------------------------
# # # Conv Block 1
# # # ------------------------------------------------------------

# # x = Conv2D(
# #     32,
# #     (3, 3),
# #     padding='same',
# #     activation='relu',
# #     name='conv1'
# # )(input1)

# # x = BatchNormalization(
# #     name='bn1'
# # )(x)

# # x = Conv2D(
# #     32,
# #     (3, 3),
# #     padding='same',
# #     activation='relu',
# #     name='conv2'
# # )(x)

# # x = BatchNormalization(
# #     name='bn2'
# # )(x)

# # x = MaxPool2D(
# #     (2, 2),
# #     name='pool1'
# # )(x)

# # x = Dropout(
# #     0.15,
# #     name='dropout1'
# # )(x)


# # # ------------------------------------------------------------
# # # Conv Block 2
# # # ------------------------------------------------------------

# # x = Conv2D(
# #     64,
# #     (3, 3),
# #     padding='same',
# #     activation='relu',
# #     name='conv3'
# # )(x)

# # x = BatchNormalization(
# #     name='bn3'
# # )(x)

# # x = Conv2D(
# #     64,
# #     (3, 3),
# #     padding='same',
# #     activation='relu',
# #     name='conv4'
# # )(x)

# # x = BatchNormalization(
# #     name='bn4'
# # )(x)

# # x = MaxPool2D(
# #     (2, 2),
# #     name='pool2'
# # )(x)

# # x = Dropout(
# #     0.20,
# #     name='dropout2'
# # )(x)


# # # ------------------------------------------------------------
# # # Conv Block 3
# # # ------------------------------------------------------------

# # x = Conv2D(
# #     128,
# #     (3, 3),
# #     padding='same',
# #     activation='relu',
# #     name='conv5'
# # )(x)

# # x = BatchNormalization(
# #     name='bn5'
# # )(x)

# # x = Conv2D(
# #     128,
# #     (3, 3),
# #     padding='same',
# #     activation='relu',
# #     name='conv6'
# # )(x)

# # x = BatchNormalization(
# #     name='bn6'
# # )(x)

# # x = MaxPool2D(
# #     (2, 2),
# #     name='pool3'
# # )(x)

# # x = Dropout(
# #     0.25,
# #     name='dropout3'
# # )(x)


# # # ------------------------------------------------------------
# # # Conv Block 4
# # # ------------------------------------------------------------

# # x = Conv2D(
# #     256,
# #     (3, 3),
# #     padding='same',
# #     activation='relu',
# #     name='conv7'
# # )(x)

# # x = BatchNormalization(
# #     name='bn7'
# # )(x)

# # x = MaxPool2D(
# #     (2, 2),
# #     name='pool4'
# # )(x)

# # x = Dropout(
# #     0.30,
# #     name='dropout4'
# # )(x)


# # # ------------------------------------------------------------
# # # Global Average Pooling
# # # ------------------------------------------------------------

# # x = GlobalAveragePooling2D(
# #     name='GAP'
# # )(x)


# # # ------------------------------------------------------------
# # # Fully Connected Layer
# # # ------------------------------------------------------------

# # x = Dense(
# #     128,
# #     activation='relu',
# #     name='dense1'
# # )(x)

# # x = BatchNormalization(
# #     name='bn_dense'
# # )(x)

# # x = Dropout(
# #     0.40,
# #     name='dropout5'
# # )(x)


# # # ------------------------------------------------------------
# # # Output Layer
# # # Binary Classification
# # # ------------------------------------------------------------

# # output1 = Dense(
# #     1,
# #     activation='sigmoid',
# #     name='output1'
# # )(x)


# # # ============================================================
# # # Functional Model
# # # ============================================================

# # model = Model(
# #     inputs=input1,
# #     outputs=output1
# # )


# # # ============================================================
# # # Model Summary
# # # ============================================================

# # model.summary()


# # # ============================================================
# # # 3. Compile
# # # ============================================================

# # optimizer = Adam(
# #     learning_rate=0.001
# # )


# # model.compile(
# #     loss='binary_crossentropy',
# #     optimizer=optimizer,
# #     metrics=['acc']
# # )


# # # ============================================================
# # # 4. Callback
# # # ============================================================

# # early_stopping = EarlyStopping(
# #     monitor='val_loss',

# #     patience=30,

# #     restore_best_weights=True,

# #     verbose=1
# # )


# # reduce_lr = ReduceLROnPlateau(
# #     monitor='val_loss',

# #     factor=0.5,

# #     patience=10,

# #     min_lr=1e-7,

# #     verbose=1
# # )


# # # ============================================================
# # # 5. Training
# # # ============================================================

# # start_time = time.time()


# # hist = model.fit(
# #     x_train,
# #     y_train,

# #     epochs=1300,

# #     batch_size=32,

# #     verbose=1,

# #     validation_split=0.2,

# #     shuffle=True,

# #     callbacks=[
# #         early_stopping,
# #         reduce_lr
# #     ]
# # )


# # end_time = time.time()


# # # ============================================================
# # # 6. Evaluation
# # # ============================================================

# # print()
# # print("=" * 70)
# # print("MODEL EVALUATE")
# # print("=" * 70)


# # loss = model.evaluate(
# #     x_test,
# #     y_test,
# #     verbose=1
# # )


# # print()
# # print("loss :", round(loss[0], 6))
# # print("acc  :", round(loss[1], 4))


# # # ============================================================
# # # 7. Prediction
# # # ============================================================

# # y_predict = model.predict(
# #     x_test,
# #     verbose=0
# # )


# # # ------------------------------------------------------------
# # # Sigmoid 확률값 확인
# # # ------------------------------------------------------------

# # print()
# # print("Prediction Probability")
# # print(
# #     np.round(
# #         y_predict[:10].reshape(-1),
# #         4
# #     )
# # )


# # # ============================================================
# # # 8. Binary Classification
# # # ============================================================
# # #
# # # 0.5 이상 → 1
# # # 0.5 미만 → 0
# # #
# # # 정확히 0.5인 경우 → 1
# # # ============================================================

# # y_predict = np.where(
# #     y_predict >= 0.5,
# #     1,
# #     0
# # ).astype(int).reshape(-1)


# # # ------------------------------------------------------------
# # # 실제 정답
# # # ------------------------------------------------------------

# # y_test = y_test.astype(int).reshape(-1)


# # # ============================================================
# # # 예측 결과 확인
# # # ============================================================

# # print()
# # print("Predicted Class")
# # print(y_predict[:20])

# # print()
# # print("Actual Class")
# # print(y_test[:20])


# # # ============================================================
# # # 9. Accuracy Score
# # # ============================================================

# # acc_score = accuracy_score(
# #     y_test,
# #     y_predict
# # )


# # print()
# # print("=" * 70)
# # print("RESULT")
# # print("=" * 70)

# # print(
# #     "accuracy_score :",
# #     round(acc_score, 4)
# # )


# # # ============================================================
# # # 10. Training Time
# # # ============================================================

# # print(
# #     "걸린시간 :",
# #     round(end_time - start_time, 2),
# #     "초"
# # )


# # # ============================================================
# # # 11. Learning Rate
# # # ============================================================

# # learning_rate = float(
# #     model.optimizer.learning_rate.numpy()
# # )


# # print(
# #     "최종 Learning Rate :",
# #     round(learning_rate, 1000)
# # )


# # # ============================================================
# # # 12. Best Epoch
# # # ============================================================

# # best_epoch = np.argmin(
# #     hist.history['val_loss']
# # ) + 1


# # best_val_loss = min(
# #     hist.history['val_loss']
# # )


# # best_val_acc = max(
# #     hist.history['val_acc']
# # )


# # print()
# # print("=" * 70)
# # print("BEST RESULT")
# # print("=" * 70)

# # print(
# #     "Best Epoch    :",
# #     best_epoch
# # )

# # print(
# #     "Best val_loss :",
# #     round(best_val_loss, 6)
# # )

# # print(
# #     "Best val_acc  :",
# #     round(best_val_acc, 4)
# # )
###############################################################################################################
#RESULT
#======================================================================
#accuracy_score : 0.9584
# 걸린시간 : 402.65 초
# 최종 Learning Rate : 6.25e-05

# ======================================================================
# BEST RESULT
# ======================================================================
# Best Epoch    : 45
# Best val_loss : 0.0983