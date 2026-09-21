

import numpy as np
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# ============================================================
# 1. npy 데이터 로드
# ============================================================
np_path = './_save/man_woman_cnn/'

x_train = np.load(np_path + 'keras47_03_x_train.npy')
y_train = np.load(np_path + 'keras47_03_y_train.npy')
x_test  = np.load(np_path + 'keras47_03_x_test.npy')
y_test  = np.load(np_path + 'keras47_03_y_test.npy')

print("x_train :", x_train.shape)   #  (10, 100, 100, 3)
print("y_train :", y_train.shape)   #  (10,)
print("x_test  :", x_test.shape)    #  (10, 100, 100, 3)
print("y_test  :", y_test.shape)    #  (10,)

print("Train MAX :", round(np.max(x_train), 4), "MIN :", round(np.min(x_train), 4))
print("Test  MAX :", round(np.max(x_test), 4),  "MIN :", round(np.min(x_test), 4))

#exit()
# ============================================================
# 2. 모델 구성 (Functional API)
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

output1 = Dense(1, activation='sigmoid', name='output1')(x)

model = Model(inputs=input1, outputs=output1)
model.summary()

# ============================================================
# 3. 컴파일
# ============================================================
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# ============================================================
# 4. 가중치 경로 설정
# ============================================================
weight_path = './_save/man_woman_weights.h5'   # 저장/불러올 경로 (폴더 미리 만들어 두세요)

# ============================================================
# ★ 여기만 바꿔서 사용하세요 ★
# True  → 학습 + 가중치 저장
# False → 가중치만 불러와서 예측
# ============================================================
TRAIN_MODE = True          # ← 처음엔 True, 학습 끝나면 False로 바꾸세요

if TRAIN_MODE:
    # -------------------- 학습 --------------------
    start_time = time.time()

    model.fit(
        x_train, y_train,
        epochs=100,
        batch_size=128,
        verbose=1,
        validation_split=0.2
    )

    end_time = time.time()
    print('걸린시간 :', round(end_time - start_time, 2), '초')

    # 가중치 저장
    model.save_weights(weight_path)
    print(f"가중치 저장 완료 → {weight_path}")

else:
    # -------------------- 가중치 불러오기 --------------------
    model.load_weights(weight_path)
    print(f"가중치 불러오기 완료 → {weight_path}")

# ============================================================
# 5. 평가 (테스트 데이터)
# ============================================================

print("\n================================ model.evaluate ================================")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print('loss :', round(test_loss, 4))
print('acc  :', round(test_acc, 4))

# 예측 + accuracy_score
y_predict_prob = model.predict(x_test, verbose=0)
y_predict = (y_predict_prob >= 0.5).astype(int).reshape(-1)
y_test_class = np.asarray(y_test).reshape(-1).astype(int)

acc_score = accuracy_score(y_test_class, y_predict)
print('accuracy_score :', round(acc_score, 4))


# ============================================================
# 6. ★ 내 사진 예측 ★
# ============================================================


print("\n================================ 내 사진 예측 ================================")

# 1) 내 사진 경로 (본인 사진으로 바꾸세요)
my_image_path = './_data/image/my_photo.png'          # ← 여기 경로 수정

# 2) 이미지 로드 + 전처리 (모델과 동일하게 100x100, grayscale, 0~1)
img = load_img(my_image_path, target_size=(100, 100), color_mode='grayscale')
img_array = img_to_array(img)                   # (100, 100, 1)
img_array = img_array / 255.0                   # 정규화
img_array = np.expand_dims(img_array, axis=0)   # (1, 100, 100, 1)

# 3) npy로 저장 (원하면)
np.save('./_data/image/my_photo.npy', img_array)
print("내 사진 npy 저장 완료 → ./_data/my_photo.npy")

# 4) 예측
pred_prob = model.predict(img_array, verbose=0)[0][0]
pred_label = 1 if pred_prob >= 0.5 else 0

print(f"예측 확률 (개일 확률) : {pred_prob:.4f}")
print("결과 →", "개 🐶" if pred_label == 1 else "고양이 🐱")

# 5) 이미지 확인
plt.imshow(img, cmap='gray')
plt.title(f"예측: {'개' if pred_label == 1 else '고양이'} ({pred_prob:.2%})")
plt.axis('off')
plt.show()

########################################    MY PHOTO PRRDICTION RESULT   #####################################
# 걸린시간 : 351.03 초
# 가중치 저장 완료 → ./_save/cat_dog_weights.h5

# ================================ model.evaluate ================================
# 157/157 [==============================] - 2s 8ms/step - loss: 2.1774 - accuracy: 0.8000 
# loss : 2.1774
# acc  : 0.8
# accuracy_score : 0.8

# ================================ 내 사진 예측 ================================
# 내 사진 npy 저장 완료 → ./_data/my_photo.npy
# 예측 확률 (개일 확률) : 0.0000
# 결과 → 고양이 🐱
################################################################################################################


# import numpy as np
# import time
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# #print(np.__version__)
# # #######################################################################################################
# # train_datagen = ImageDataGenerator(
# #     rescale=1./255,   #. 부동소숫점 형변환
# #     horizontal_flip=True,  #이미지를 좌우 반전합니다.
# #     vertical_flip=True,    #이미지를 상하 반전합니다.
# #     width_shift_range=0.1,
# #     height_shift_range=0.1,
# #     rotation_range=5,
# #     zoom_range=1.2,
# #     shear_range=0.7, # 전단 변환(Shear Transformation)**을 적용하는 옵션
# #     fill_mode='nearest'
# # )
# # test_datagen = ImageDataGenerator(
# #     rescale=1./255,
# # )

# # path_train = './_data/image/cat_dog/training_set/'   #  Found 8005 images belonging to 2 classes.
# # path_test = './_data/image/cat_dog/test_set/'        #  Found 2023 images belonging to 2 classes.

# # xy_train = train_datagen.flow_from_directory(
# #     path_train, #경로
# #     target_size=(100,100),
# #     batch_size=5000,
# #     class_mode='binary', # 이진분류
# #     color_mode='grayscale', #흑백
# #     shuffle=False,
# # )
# # #Found 160 images belonging to 2 classes.

# # xy_test = train_datagen.flow_from_directory(
# #     path_test,
# #     target_size=(100,100),
# #     batch_size=10,
# #     class_mode='binary', # 이진분류
# #     color_mode='grayscale', #흑백
# #     shuffle=False,
# # )
# # #Found 120 images belonging to 2 classes.


# # print(xy_train)
# # #<keras.preprocessing.image.DirectoryIterator object at 0x000001962E757FA0>
# # print(xy_test)
# # #<keras.preprocessing.image.DirectoryIterator object at 0x000001962E7579D0>
# # #exit()
# # print(xy_train.next()) # Iterator 첫번째를 보여줘?
# # print(xy_train.next()) # 두번째 Iterator 출력해줘?

# # # print(xy_train[0])
# # # print(xy_train[1])
# # # print(xy_train[2])

# # #print(xy_train[0][0]  # 첫번째 배치의 X 데이터가 되겠지요.
# # #print(xy_train[0][1]  # 첫번째 배치의 Y 데이터가 되겠지요.


# # #exit()
# # # ============================================================
# # # Generator에서 데이터 추출
# # # ============================================================

# # x_train = xy_train[0][0]
# # y_train = xy_train[0][1]

# # x_test = xy_test[0][0]
# # y_test = xy_test[0][1]

# # ######################################################################################################
# # # ============================================================
# # # Shape 확인
# # # ============================================================

# # print()
# # print("x_train :", x_train.shape)    #(160, 150, 150, 1)
# # print("y_train :", y_train.shape)

# # print("x_test  :", x_test.shape)
# # print("y_test  :", y_test.shape)

# # #exit()
# # # np_path = './_data/kaggle_cat_dog_npy/'
# # # np.save(np_path + 'keras45_01_x_train.npy', arr=x_train[0][0])
# # # np.save(np_path + 'keras45_01_y_train.npy', arr=x_train[0][1])

# # # np.save(np_path + 'keras45_01_x_test.npy', arr=x_test[0][0])
# # # np.save(np_path + 'keras45_01_y_test.npy', arr=x_train[0][1])
# ################################################################################################
# ####################################################### np.load()  ##############################

# np_path = './_data/kaggle_cat_dog_npy/'


# x_train= np.load(np_path + 'keras45_03_x_train.npy')
# y_train= np.load(np_path + 'keras45_03_y_train.npy')
# x_test= np.load(np_path + 'keras45_03_x_test.npy')
# y_test= np.load(np_path + 'keras45_03_y_test.npy')

# ###################################################### np.load ()  ###############################


# #exit()

# # ============================================================
# # Shape 확인
# # ============================================================

# print()
# print("x_train :", x_train.shape) # (5000, 100, 100, 1)
# print("y_train :", y_train.shape) # (5000,)

# print("x_test  :", x_test.shape)  #  (5000, 100, 100, 1)
# print("y_test  :", y_test.shape)  #  (5000,)

# #exit()
# # ============================================================
# # Scaling 확인
# # ============================================================

# print()
# print("Train MAX :", round(np.max(x_train), 4))
# print("Train MIN :", round(np.min(x_train), 4))

# print("Test MAX  :", round(np.max(x_test), 4))
# print("Test MIN  :", round(np.min(x_test), 4))

# #Train MAX : 1.0
# #Train MIN : 0.0
# #Test MAX  : 1.0
# #Test MIN  : 0.0118
# #exit()



# # ============================================================
# # 2. CNN 함수형 모델 구성 - 수평형
# # ============================================================

# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Conv2D, MaxPool2D
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import GlobalAveragePooling2D
# from tensorflow.keras.layers import Dense, Dropout


# # ============================================================
# # 1. Input Layer
# # ============================================================

# input1 = Input(shape=(100, 100, 1), name='input1')


# # ============================================================
# # 2. Conv Block 1
# # ============================================================

# conv1_out = Conv2D(32, (3, 3), padding='same',
#                    activation='relu', name='conv1')(input1)
# bn1_out = BatchNormalization(name='bn1')(conv1_out)

# conv2_out = Conv2D(32, (3, 3), padding='same',
#                    activation='relu', name='conv2')(bn1_out)
# bn2_out = BatchNormalization(name='bn2')(conv2_out)

# pool1_out = MaxPool2D((2, 2), name='pool1')(bn2_out)
# dropout1_out = Dropout(0.15, name='dropout1')(pool1_out)


# # ============================================================
# # 3. Conv Block 2
# # ============================================================

# conv3_out = Conv2D(64, (3, 3), padding='same',
#                    activation='relu', name='conv3')(dropout1_out)
# bn3_out = BatchNormalization(name='bn3')(conv3_out)

# conv4_out = Conv2D(64, (3, 3), padding='same',
#                    activation='relu', name='conv4')(bn3_out)
# bn4_out = BatchNormalization(name='bn4')(conv4_out)

# pool2_out = MaxPool2D((2, 2), name='pool2')(bn4_out)
# dropout2_out = Dropout(0.20, name='dropout2')(pool2_out)


# # ============================================================
# # 4. Conv Block 3
# # ============================================================

# conv5_out = Conv2D(128, (3, 3), padding='same',
#                    activation='relu', name='conv5')(dropout2_out)
# bn5_out = BatchNormalization(name='bn5')(conv5_out)

# conv6_out = Conv2D(128, (3, 3), padding='same',
#                    activation='relu', name='conv6')(bn5_out)
# bn6_out = BatchNormalization(name='bn6')(conv6_out)

# pool3_out = MaxPool2D((2, 2), name='pool3')(bn6_out)
# dropout3_out = Dropout(0.25, name='dropout3')(pool3_out)


# # ============================================================
# # 5. Conv Block 4
# # ============================================================

# conv7_out = Conv2D(256, (3, 3), padding='same',
#                    activation='relu', name='conv7')(dropout3_out)
# bn7_out = BatchNormalization(name='bn7')(conv7_out)

# pool4_out = MaxPool2D((2, 2), name='pool4')(bn7_out)
# dropout4_out = Dropout(0.30, name='dropout4')(pool4_out)


# # ============================================================
# # 6. Global Average Pooling
# # ============================================================

# gap_out = GlobalAveragePooling2D(name='GAP')(dropout4_out)


# # ============================================================
# # 7. Fully Connected Layer
# # ============================================================

# dense1_out = Dense(128, activation='relu',
#                    name='dense1')(gap_out)
# bn_dense_out = BatchNormalization(name='bn_dense')(dense1_out)
# dropout5_out = Dropout(0.40, name='dropout5')(bn_dense_out)


# # ============================================================
# # 8. Output Layer - Binary Classification
# # ============================================================

# output1 = Dense(1, activation='sigmoid',
#                 name='output1')(dropout5_out)


# # ============================================================
# # 9. Functional Model
# # ============================================================

# model = Model(inputs=input1, outputs=output1)


# # ============================================================
# # 10. Model Summary
# # ============================================================

# model.summary()




# # ============================================================
# # 3. 컴파일
# # ============================================================

# model.compile(
#     loss='binary_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy']
# )


# # ============================================================
# # 3-1. 훈련
# # ============================================================

# start_time = time.time()

# model.fit(
#     x_train,
#     y_train,
#     epochs=100,
#     batch_size=128,
#     verbose=1,
#     validation_split=0.2
# )

# end_time = time.time()


# # ============================================================
# # 4. 평가
# # ============================================================

# print("================================ model.evaluate ================================")

# test_loss, test_acc = model.evaluate(
#     x_test,
#     y_test,
#     verbose=1
# )

# print('loss :', round(test_loss, 4))
# print('acc  :', round(test_acc, 4))


# # ============================================================
# # 5. 예측
# # ============================================================

# y_predict_prob = model.predict(
#     x_test,
#     verbose=0
# )


# # ============================================================
# # 5-1. 확률 → 0 또는 1
# #     0.5 이상 → 1
# #     0.5 미만 → 0
# # ============================================================

# y_predict = (y_predict_prob >= 0.5).astype(int).reshape(-1)


# # ============================================================
# # 5-2. 실제 정답 데이터 형태 정리
# # ============================================================

# y_test_class = np.asarray(y_test).reshape(-1).astype(int)


# # ============================================================
# # 5-3. Accuracy 계산
# # ============================================================

# acc_score = accuracy_score(
#     y_test_class,
#     y_predict
# )

# print('accuracy_score :', round(acc_score, 4))


# # ============================================================
# # 6. 실행 시간
# # ============================================================

# print(
#     '걸린시간 :',
#     round(end_time - start_time, 2),
#     '초'
# )
# #####################################################################################################
# #loss : 0.0
# #acc  : 1.0

# #######################################################  loaded ######################################
# # loss : 2.1847
# # acc  : 0.8000
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
# np_ath = './_data/image/horse-human/'

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

# #exit()
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
# print("x_train :", x_train.shape) # (5000, 100, 100, 1)
# print("y_train :", y_train.shape) # (5000,)

# print("x_test  :", x_test.shape)  #  (10, 100, 100, 1)
# print("y_test  :", y_test.shape)  #  (10,)

# ###################################################################################################################
# x_train = xy_train[0][0]
# y_train = xy_train[0][1]
# x_test = xy_test[0][0]
# y_test = xy_test[0][1]

# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

# ############################################    np.save( )                    ########################################
# np_path = './_data/hourse-human_npy/'             #
# np.save(np_path + 'keras47_03_x_train.npy' , arr = xy_train[0][0])  # arr = x_train 도가능 
# np.save(np_path + 'keras47_03_y_train.npy' , arr = xy_train[0][1])  # arr = y_train 도가능
# np.save(np_path + 'keras47_03_x_test.npy' , arr = xy_train[0][0])   # arr = x_test 도가능
# np.save(np_path + 'keras47_03_y_test.npy' , arr = xy_train[0][1])   # arr = y_test 도가능
# ######################################################################################################################


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

# output1 = Dense(1, activation='sigmoid', name='output1')(x)

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
# import numpy as np
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import numpy as np
# import time

# from keras.preprocessing.image import ImageDataGenerator

# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input
# from tensorflow.keras.layers import Conv2D, Dense, Dropout
# from tensorflow.keras.layers import MaxPool2D, GlobalAveragePooling2D
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# from sklearn.metrics import accuracy_score
# print(np.__version__)

# # ============================================================
# # 1. ImageDataGenerator 설정
# # ============================================================
# train_datagen = ImageDataGenerator(
#     rescale=1./255,              # 0~1 정규화
#     horizontal_flip=True,        # 좌우 반전
#     vertical_flip=True,          # 상하 반전
#     width_shift_range=0.1,       # 가로 이동
#     height_shift_range=0.1,      # 세로 이동
#     rotation_range=5,            # 회전
#     zoom_range=1.2,              # 확대/축소
#     shear_range=0.7,             # 전단 변환
#     fill_mode='nearest',
#     validation_split=0.2         # 20%를 검증용으로 분리
# )

# test_datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2         # 같은 비율로 맞춰줌
# )

# # ============================================================
# # 2. 데이터 경로
# # ============================================================
# path = './_data/image/horse-human/'   # horses / humans 폴더가 들어있는 경로

# # ============================================================
# # 3. flow_from_directory로 데이터 생성
# # ============================================================
# xy_train = train_datagen.flow_from_directory(
#     path,
#     target_size=(100, 100),      # 원하는 크기로 리사이즈
#     batch_size=10,
#     class_mode='binary',         # 이진 분류 (horse / human)
#     color_mode='rgb',            # 컬러 이미지 (grayscale이 아님)
#     shuffle=True,
#     subset='training',           # 80% 훈련 데이터
#     seed=42
# )

# xy_test = test_datagen.flow_from_directory(
#     path,
#     target_size=(100, 100),
#     batch_size=10,
#     class_mode='binary',
#     color_mode='rgb',
#     shuffle=False,
#     subset='validation',         # 20% 검증 데이터
#     seed=42
# )

# # ============================================================
# # 4. 확인용 출력
# # ============================================================
# print(xy_train)
# print(xy_test)

# print(xy_train[0][0].shape)   # (10, 100, 100, 3)
# print(xy_train[0][1].shape)   # (10,)

# print(type(xy_train))         # <class 'keras...DirectoryIterator'>
# print(type(xy_train[0]))      # <class 'tuple'>
# print(type(xy_train[0][0]))   # <class 'numpy.ndarray'>
# print(type(xy_train[0][1]))   # <class 'numpy.ndarray'>

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
# # 2. CNN 함수형 모델 구성 - 성능 개선
# # ============================================================

# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import (
#     Input, Conv2D, MaxPool2D,
#     BatchNormalization, GlobalAveragePooling2D,
#     Dense, Dropout
# )


# # ------------------------------------------------------------
# # Input Layer
# # ------------------------------------------------------------

# input1 = Input(
#     shape=(100, 100, 3),
#     name='input1'
# )


# # ------------------------------------------------------------
# # Conv Block 1
# # ------------------------------------------------------------

# x = Conv2D(
#     32,
#     (3, 3),
#     padding='same',
#     activation='relu',
#     name='conv1'
# )(input1)

# x = BatchNormalization(
#     name='bn1'
# )(x)

# x = Conv2D(
#     32,
#     (3, 3),
#     padding='same',
#     activation='relu',
#     name='conv2'
# )(x)

# x = BatchNormalization(
#     name='bn2'
# )(x)

# x = MaxPool2D(
#     (2, 2),
#     name='pool1'
# )(x)

# x = Dropout(
#     0.15,
#     name='dropout1'
# )(x)


# # ------------------------------------------------------------
# # Conv Block 2
# # ------------------------------------------------------------

# x = Conv2D(
#     64,
#     (3, 3),
#     padding='same',
#     activation='relu',
#     name='conv3'
# )(x)

# x = BatchNormalization(
#     name='bn3'
# )(x)

# x = Conv2D(
#     64,
#     (3, 3),
#     padding='same',
#     activation='relu',
#     name='conv4'
# )(x)

# x = BatchNormalization(
#     name='bn4'
# )(x)

# x = MaxPool2D(
#     (2, 2),
#     name='pool2'
# )(x)

# x = Dropout(
#     0.20,
#     name='dropout2'
# )(x)


# # ------------------------------------------------------------
# # Conv Block 3
# # ------------------------------------------------------------

# x = Conv2D(
#     128,
#     (3, 3),
#     padding='same',
#     activation='relu',
#     name='conv5'
# )(x)

# x = BatchNormalization(
#     name='bn5'
# )(x)

# x = Conv2D(
#     128,
#     (3, 3),
#     padding='same',
#     activation='relu',
#     name='conv6'
# )(x)

# x = BatchNormalization(
#     name='bn6'
# )(x)

# x = MaxPool2D(
#     (2, 2),
#     name='pool3'
# )(x)

# x = Dropout(
#     0.25,
#     name='dropout3'
# )(x)


# # ------------------------------------------------------------
# # Conv Block 4
# # ------------------------------------------------------------

# x = Conv2D(
#     256,
#     (3, 3),
#     padding='same',
#     activation='relu',
#     name='conv7'
# )(x)

# x = BatchNormalization(
#     name='bn7'
# )(x)

# x = MaxPool2D(
#     (2, 2),
#     name='pool4'
# )(x)

# x = Dropout(
#     0.30,
#     name='dropout4'
# )(x)


# # ------------------------------------------------------------
# # Global Average Pooling
# # ------------------------------------------------------------

# x = GlobalAveragePooling2D(
#     name='GAP'
# )(x)


# # ------------------------------------------------------------
# # Fully Connected Layer
# # ------------------------------------------------------------

# x = Dense(
#     128,
#     activation='relu',
#     name='dense1'
# )(x)

# x = BatchNormalization(
#     name='bn_dense'
# )(x)

# x = Dropout(
#     0.40,
#     name='dropout5'
# )(x)


# # ------------------------------------------------------------
# # Output Layer
# # Binary Classification
# # ------------------------------------------------------------

# output1 = Dense(
#     1,
#     activation='sigmoid',
#     name='output1'
# )(x)


# # ============================================================
# # Functional Model
# # ============================================================

# model = Model(
#     inputs=input1,
#     outputs=output1
# )


# # ============================================================
# # Model Summary
# # ============================================================

# model.summary()


# # ============================================================
# # 3. Compile
# # ============================================================

# optimizer = Adam(
#     learning_rate=0.001
# )


# model.compile(
#     loss='binary_crossentropy',
#     optimizer=optimizer,
#     metrics=['acc']
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
# # 5. Training
# # ============================================================

# start_time = time.time()


# hist = model.fit(
#     x_train,
#     y_train,

#     epochs=1300,

#     batch_size=32,

#     verbose=1,

#     validation_split=0.2,

#     shuffle=True,

#     callbacks=[
#         early_stopping,
#         reduce_lr
#     ]
# )


# end_time = time.time()


# # ============================================================
# # 6. Evaluation
# # ============================================================

# print()
# print("=" * 70)
# print("MODEL EVALUATE")
# print("=" * 70)


# loss = model.evaluate(
#     x_test,
#     y_test,
#     verbose=1
# )


# print()
# print("loss :", round(loss[0], 6))
# print("acc  :", round(loss[1], 4))


# # ============================================================
# # 7. Prediction
# # ============================================================

# y_predict = model.predict(
#     x_test,
#     verbose=0
# )


# # ------------------------------------------------------------
# # Sigmoid 확률값 확인
# # ------------------------------------------------------------

# print()
# print("Prediction Probability")
# print(
#     np.round(
#         y_predict[:10].reshape(-1),
#         4
#     )
# )


# # ============================================================
# # 8. Binary Classification
# # ============================================================
# #
# # 0.5 이상 → 1
# # 0.5 미만 → 0
# #
# # 정확히 0.5인 경우 → 1
# # ============================================================

# y_predict = np.where(
#     y_predict >= 0.5,
#     1,
#     0
# ).astype(int).reshape(-1)


# # ------------------------------------------------------------
# # 실제 정답
# # ------------------------------------------------------------

# y_test = y_test.astype(int).reshape(-1)


# # ============================================================
# # 예측 결과 확인
# # ============================================================

# print()
# print("Predicted Class")
# print(y_predict[:20])

# print()
# print("Actual Class")
# print(y_test[:20])


# # ============================================================
# # 9. Accuracy Score
# # ============================================================

# acc_score = accuracy_score(
#     y_test,
#     y_predict
# )


# print()
# print("=" * 70)
# print("RESULT")
# print("=" * 70)

# print(
#     "accuracy_score :",
#     round(acc_score, 4)
# )


# # ============================================================
# # 10. Training Time
# # ============================================================

# print(
#     "걸린시간 :",
#     round(end_time - start_time, 2),
#     "초"
# )


# # ============================================================
# # 11. Learning Rate
# # ============================================================

# learning_rate = float(
#     model.optimizer.learning_rate.numpy()
# )


# print(
#     "최종 Learning Rate :",
#     round(learning_rate, 1000)
# )


# # ============================================================
# # 12. Best Epoch
# # ============================================================

# best_epoch = np.argmin(
#     hist.history['val_loss']
# ) + 1


# best_val_loss = min(
#     hist.history['val_loss']
# )


# best_val_acc = max(
#     hist.history['val_acc']
# )


# print()
# print("=" * 70)
# print("BEST RESULT")
# print("=" * 70)

# print(
#     "Best Epoch    :",
#     best_epoch
# )

# print(
#     "Best val_loss :",
#     round(best_val_loss, 6)
# )

# print(
#     "Best val_acc  :",
#     round(best_val_acc, 4)
# )
###############################################################################################################
# accuracy_score : 0.8976
# 걸린시간 : 366.19 초
# 최종 Learning Rate : 3.125e-05

# ======================================================================
# BEST RESULT
# ======================================================================
# Best Epoch    : 47
# Best val_loss : 0.221566
# Best val_acc  : 0.9024
##############################################################################################################

"""
개고양이 가중치를 가져와서 모델완성
데이터는 개 고양이 npy데이터 사용
내사진도 npy 불러와서 predict만 하면 되겠지요.
"""
"""
개고양이 가중치를 가져와서 모델완성
데이터는 개 고양이 npy데이터 사용
내사진도 npy 불러와서 predict만 하면 되겠지요.
"""

import numpy as np
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# ============================================================
# 1. npy 데이터 로드
# ============================================================
np_path = './_data/kaggle_cat_dog_npy/'

x_train = np.load(np_path + 'keras45_03_x_train.npy')
y_train = np.load(np_path + 'keras45_03_y_train.npy')
x_test  = np.load(np_path + 'keras45_03_x_test.npy')
y_test  = np.load(np_path + 'keras45_03_y_test.npy')

print("x_train :", x_train.shape)   # (5000, 100, 100, 1)
print("y_train :", y_train.shape)   # (5000,)
print("x_test  :", x_test.shape)
print("y_test  :", y_test.shape)

print("Train MAX :", round(np.max(x_train), 4), "MIN :", round(np.min(x_train), 4))
print("Test  MAX :", round(np.max(x_test), 4),  "MIN :", round(np.min(x_test), 4))

# ============================================================
# 2. 모델 구성 (Functional API)
# ============================================================
input1 = Input(shape=(100, 100, 1), name='input1')

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

output1 = Dense(1, activation='sigmoid', name='output1')(x)

model = Model(inputs=input1, outputs=output1)
model.summary()

# ============================================================
# 3. 컴파일
# ============================================================
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# ============================================================
# 4. 가중치 경로 설정
# ============================================================
weight_path = './_save/cat_dog_weights.h5'   # 저장/불러올 경로 (폴더 미리 만들어 두세요)

# ============================================================
# ★ 여기만 바꿔서 사용하세요 ★
# True  → 학습 + 가중치 저장
# False → 가중치만 불러와서 예측
# ============================================================
TRAIN_MODE = True          # ← 처음엔 True, 학습 끝나면 False로 바꾸세요

if TRAIN_MODE:
    # -------------------- 학습 --------------------
    start_time = time.time()

    model.fit(
        x_train, y_train,
        epochs=100,
        batch_size=128,
        verbose=1,
        validation_split=0.2
    )

    end_time = time.time()
    print('걸린시간 :', round(end_time - start_time, 2), '초')

    # 가중치 저장
    model.save_weights(weight_path)
    print(f"가중치 저장 완료 → {weight_path}")

else:
    # -------------------- 가중치 불러오기 --------------------
    model.load_weights(weight_path)
    print(f"가중치 불러오기 완료 → {weight_path}")

# ============================================================
# 5. 평가 (테스트 데이터)
# ============================================================
print("\n================================ model.evaluate ================================")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print('loss :', round(test_loss, 4))
print('acc  :', round(test_acc, 4))

# 예측 + accuracy_score
y_predict_prob = model.predict(x_test, verbose=0)
y_predict = (y_predict_prob >= 0.5).astype(int).reshape(-1)
y_test_class = np.asarray(y_test).reshape(-1).astype(int)

acc_score = accuracy_score(y_test_class, y_predict)
print('accuracy_score :', round(acc_score, 4))

# ============================================================
# 6. ★ 내 사진 예측 ★
# ============================================================
print("\n================================ 내 사진 예측 ================================")

# 1) 내 사진 경로 (본인 사진으로 바꾸세요)
my_image_path = './_data/image/my_photo.png'          # ← 여기 경로 수정

# 2) 이미지 로드 + 전처리 (모델과 동일하게 100x100, grayscale, 0~1)
img = load_img(my_image_path, target_size=(100, 100), color_mode='grayscale')
img_array = img_to_array(img)                   # (100, 100, 1)
img_array = img_array / 255.0                   # 정규화
img_array = np.expand_dims(img_array, axis=0)   # (1, 100, 100, 1)

# 3) npy로 저장 (원하면)
np.save('./_data/image/my_photo.npy', img_array)
print("내 사진 npy 저장 완료 → ./_data/my_photo.npy")

# 4) 예측
pred_prob = model.predict(img_array, verbose=0)[0][0]
pred_label = 1 if pred_prob >= 0.5 else 0

print(f"예측 확률 (개일 확률) : {pred_prob:.4f}")
print("결과 →", "개 🐶" if pred_label == 1 else "고양이 🐱")

# 5) 이미지 확인
plt.imshow(img, cmap='gray')
plt.title(f"예측: {'개' if pred_label == 1 else '고양이'} ({pred_prob:.2%})")
plt.axis('off')
plt.show()

########################################    MY PHOTO PRRDICTION RESULT   #####################################
# 걸린시간 : 351.03 초
# 가중치 저장 완료 → ./_save/cat_dog_weights.h5

# ================================ model.evaluate ================================
# 157/157 [==============================] - 2s 8ms/step - loss: 2.1774 - accuracy: 0.8000 
# loss : 2.1774
# acc  : 0.8
# accuracy_score : 0.8

# ================================ 내 사진 예측 ================================
# 내 사진 npy 저장 완료 → ./_data/my_photo.npy
# 예측 확률 (개일 확률) : 0.0000
# 결과 → 고양이 🐱
################################################################################################################


# import numpy as np
# import time
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# #print(np.__version__)
# # #######################################################################################################
# # train_datagen = ImageDataGenerator(
# #     rescale=1./255,   #. 부동소숫점 형변환
# #     horizontal_flip=True,  #이미지를 좌우 반전합니다.
# #     vertical_flip=True,    #이미지를 상하 반전합니다.
# #     width_shift_range=0.1,
# #     height_shift_range=0.1,
# #     rotation_range=5,
# #     zoom_range=1.2,
# #     shear_range=0.7, # 전단 변환(Shear Transformation)**을 적용하는 옵션
# #     fill_mode='nearest'
# # )
# # test_datagen = ImageDataGenerator(
# #     rescale=1./255,
# # )

# # path_train = './_data/image/cat_dog/training_set/'   #  Found 8005 images belonging to 2 classes.
# # path_test = './_data/image/cat_dog/test_set/'        #  Found 2023 images belonging to 2 classes.

# # xy_train = train_datagen.flow_from_directory(
# #     path_train, #경로
# #     target_size=(100,100),
# #     batch_size=5000,
# #     class_mode='binary', # 이진분류
# #     color_mode='grayscale', #흑백
# #     shuffle=False,
# # )
# # #Found 160 images belonging to 2 classes.

# # xy_test = train_datagen.flow_from_directory(
# #     path_test,
# #     target_size=(100,100),
# #     batch_size=10,
# #     class_mode='binary', # 이진분류
# #     color_mode='grayscale', #흑백
# #     shuffle=False,
# # )
# # #Found 120 images belonging to 2 classes.


# # print(xy_train)
# # #<keras.preprocessing.image.DirectoryIterator object at 0x000001962E757FA0>
# # print(xy_test)
# # #<keras.preprocessing.image.DirectoryIterator object at 0x000001962E7579D0>
# # #exit()
# # print(xy_train.next()) # Iterator 첫번째를 보여줘?
# # print(xy_train.next()) # 두번째 Iterator 출력해줘?

# # # print(xy_train[0])
# # # print(xy_train[1])
# # # print(xy_train[2])

# # #print(xy_train[0][0]  # 첫번째 배치의 X 데이터가 되겠지요.
# # #print(xy_train[0][1]  # 첫번째 배치의 Y 데이터가 되겠지요.


# # #exit()
# # # ============================================================
# # # Generator에서 데이터 추출
# # # ============================================================

# # x_train = xy_train[0][0]
# # y_train = xy_train[0][1]

# # x_test = xy_test[0][0]
# # y_test = xy_test[0][1]

# # ######################################################################################################
# # # ============================================================
# # # Shape 확인
# # # ============================================================

# # print()
# # print("x_train :", x_train.shape)    #(160, 150, 150, 1)
# # print("y_train :", y_train.shape)

# # print("x_test  :", x_test.shape)
# # print("y_test  :", y_test.shape)

# # #exit()
# # # np_path = './_data/kaggle_cat_dog_npy/'
# # # np.save(np_path + 'keras45_01_x_train.npy', arr=x_train[0][0])
# # # np.save(np_path + 'keras45_01_y_train.npy', arr=x_train[0][1])

# # # np.save(np_path + 'keras45_01_x_test.npy', arr=x_test[0][0])
# # # np.save(np_path + 'keras45_01_y_test.npy', arr=x_train[0][1])
# ################################################################################################
# ####################################################### np.load()  ##############################

# np_path = './_data/kaggle_cat_dog_npy/'


# x_train= np.load(np_path + 'keras45_03_x_train.npy')
# y_train= np.load(np_path + 'keras45_03_y_train.npy')
# x_test= np.load(np_path + 'keras45_03_x_test.npy')
# y_test= np.load(np_path + 'keras45_03_y_test.npy')

# ###################################################### np.load ()  ###############################


# #exit()

# # ============================================================
# # Shape 확인
# # ============================================================

# print()
# print("x_train :", x_train.shape) # (5000, 100, 100, 1)
# print("y_train :", y_train.shape) # (5000,)

# print("x_test  :", x_test.shape)  #  (5000, 100, 100, 1)
# print("y_test  :", y_test.shape)  #  (5000,)

# #exit()
# # ============================================================
# # Scaling 확인
# # ============================================================

# print()
# print("Train MAX :", round(np.max(x_train), 4))
# print("Train MIN :", round(np.min(x_train), 4))

# print("Test MAX  :", round(np.max(x_test), 4))
# print("Test MIN  :", round(np.min(x_test), 4))

# #Train MAX : 1.0
# #Train MIN : 0.0
# #Test MAX  : 1.0
# #Test MIN  : 0.0118
# #exit()



# # ============================================================
# # 2. CNN 함수형 모델 구성 - 수평형
# # ============================================================

# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Conv2D, MaxPool2D
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import GlobalAveragePooling2D
# from tensorflow.keras.layers import Dense, Dropout


# # ============================================================
# # 1. Input Layer
# # ============================================================

# input1 = Input(shape=(100, 100, 1), name='input1')


# # ============================================================
# # 2. Conv Block 1
# # ============================================================

# conv1_out = Conv2D(32, (3, 3), padding='same',
#                    activation='relu', name='conv1')(input1)
# bn1_out = BatchNormalization(name='bn1')(conv1_out)

# conv2_out = Conv2D(32, (3, 3), padding='same',
#                    activation='relu', name='conv2')(bn1_out)
# bn2_out = BatchNormalization(name='bn2')(conv2_out)

# pool1_out = MaxPool2D((2, 2), name='pool1')(bn2_out)
# dropout1_out = Dropout(0.15, name='dropout1')(pool1_out)


# # ============================================================
# # 3. Conv Block 2
# # ============================================================

# conv3_out = Conv2D(64, (3, 3), padding='same',
#                    activation='relu', name='conv3')(dropout1_out)
# bn3_out = BatchNormalization(name='bn3')(conv3_out)

# conv4_out = Conv2D(64, (3, 3), padding='same',
#                    activation='relu', name='conv4')(bn3_out)
# bn4_out = BatchNormalization(name='bn4')(conv4_out)

# pool2_out = MaxPool2D((2, 2), name='pool2')(bn4_out)
# dropout2_out = Dropout(0.20, name='dropout2')(pool2_out)


# # ============================================================
# # 4. Conv Block 3
# # ============================================================

# conv5_out = Conv2D(128, (3, 3), padding='same',
#                    activation='relu', name='conv5')(dropout2_out)
# bn5_out = BatchNormalization(name='bn5')(conv5_out)

# conv6_out = Conv2D(128, (3, 3), padding='same',
#                    activation='relu', name='conv6')(bn5_out)
# bn6_out = BatchNormalization(name='bn6')(conv6_out)

# pool3_out = MaxPool2D((2, 2), name='pool3')(bn6_out)
# dropout3_out = Dropout(0.25, name='dropout3')(pool3_out)


# # ============================================================
# # 5. Conv Block 4
# # ============================================================

# conv7_out = Conv2D(256, (3, 3), padding='same',
#                    activation='relu', name='conv7')(dropout3_out)
# bn7_out = BatchNormalization(name='bn7')(conv7_out)

# pool4_out = MaxPool2D((2, 2), name='pool4')(bn7_out)
# dropout4_out = Dropout(0.30, name='dropout4')(pool4_out)


# # ============================================================
# # 6. Global Average Pooling
# # ============================================================

# gap_out = GlobalAveragePooling2D(name='GAP')(dropout4_out)


# # ============================================================
# # 7. Fully Connected Layer
# # ============================================================

# dense1_out = Dense(128, activation='relu',
#                    name='dense1')(gap_out)
# bn_dense_out = BatchNormalization(name='bn_dense')(dense1_out)
# dropout5_out = Dropout(0.40, name='dropout5')(bn_dense_out)


# # ============================================================
# # 8. Output Layer - Binary Classification
# # ============================================================

# output1 = Dense(1, activation='sigmoid',
#                 name='output1')(dropout5_out)


# # ============================================================
# # 9. Functional Model
# # ============================================================

# model = Model(inputs=input1, outputs=output1)


# # ============================================================
# # 10. Model Summary
# # ============================================================

# model.summary()




# # ============================================================
# # 3. 컴파일
# # ============================================================

# model.compile(
#     loss='binary_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy']
# )


# # ============================================================
# # 3-1. 훈련
# # ============================================================

# start_time = time.time()

# model.fit(
#     x_train,
#     y_train,
#     epochs=100,
#     batch_size=128,
#     verbose=1,
#     validation_split=0.2
# )

# end_time = time.time()


# # ============================================================
# # 4. 평가
# # ============================================================

# print("================================ model.evaluate ================================")

# test_loss, test_acc = model.evaluate(
#     x_test,
#     y_test,
#     verbose=1
# )

# print('loss :', round(test_loss, 4))
# print('acc  :', round(test_acc, 4))


# # ============================================================
# # 5. 예측
# # ============================================================

# y_predict_prob = model.predict(
#     x_test,
#     verbose=0
# )


# # ============================================================
# # 5-1. 확률 → 0 또는 1
# #     0.5 이상 → 1
# #     0.5 미만 → 0
# # ============================================================

# y_predict = (y_predict_prob >= 0.5).astype(int).reshape(-1)


# # ============================================================
# # 5-2. 실제 정답 데이터 형태 정리
# # ============================================================

# y_test_class = np.asarray(y_test).reshape(-1).astype(int)


# # ============================================================
# # 5-3. Accuracy 계산
# # ============================================================

# acc_score = accuracy_score(
#     y_test_class,
#     y_predict
# )

# print('accuracy_score :', round(acc_score, 4))


# # ============================================================
# # 6. 실행 시간
# # ============================================================

# print(
#     '걸린시간 :',
#     round(end_time - start_time, 2),
#     '초'
# )
# #####################################################################################################
# #loss : 0.0
# #acc  : 1.0

# #######################################################  loaded ######################################
# # loss : 2.1847
# # acc  : 0.8000