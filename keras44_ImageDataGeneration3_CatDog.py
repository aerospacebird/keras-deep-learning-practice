import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
print(np.__version__)

train_datagen = ImageDataGenerator(

    rescale=1./255,          # 픽셀값을 0~255 → 0~1로 정규화

    horizontal_flip=True,    # 이미지를 좌우로 반전

    vertical_flip=True,      # 이미지를 상하로 반전

    width_shift_range=0.1,   # 이미지 너비의 10% 범위에서 좌우 이동

    height_shift_range=0.1,  # 이미지 높이의 10% 범위에서 상하 이동

    rotation_range=5,        # -5° ~ +5° 범위에서 회전

    zoom_range=1.2,          # 이미지를 확대/축소하여 데이터 증강

    shear_range=0.7,         # 이미지를 기울이는 전단 변환 적용

    fill_mode='nearest'      # 변환 후 빈 공간을 가장 가까운 픽셀로 채움
)


test_datagen = ImageDataGenerator(

    rescale=1./255           # 테스트 이미지의 픽셀값을 0~1로 정규화

)

path_train = './_data/image/cat_dog/training_set/'   #  Found 8005 images belonging to 2 classes.
path_test = './_data/image/cat_dog/test_set/'        #  Found 2023 images belonging to 2 classes.

xy_train = train_datagen.flow_from_directory(
    path_train, #경로
    target_size=(100,100),
    batch_size=5000,
    class_mode='binary', # 이진분류
    color_mode='grayscale', #흑백
    shuffle=False,
)
#Found 160 images belonging to 2 classes.

xy_test = train_datagen.flow_from_directory(
    path_test,
    target_size=(100,100),
    batch_size=10,
    class_mode='binary', # 이진분류
    color_mode='grayscale', #흑백
    shuffle=False,
)
#Found 120 images belonging to 2 classes.


print(xy_train)
#<keras.preprocessing.image.DirectoryIterator object at 0x000001962E757FA0>
print(xy_test)
#<keras.preprocessing.image.DirectoryIterator object at 0x000001962E7579D0>
#exit()
print(xy_train.next()) # Iterator 첫번째를 보여줘?
print(xy_train.next()) # 두번째 Iterator 출력해줘?

# print(xy_train[0])
# print(xy_train[1])
# print(xy_train[2])

#print(xy_train[0][0]  # 첫번째 배치의 X 데이터가 되겠지요.
#print(xy_train[0][1]  # 첫번째 배치의 Y 데이터가 되겠지요.


#exit()
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
print("x_train :", x_train.shape) # (5000, 100, 100, 1)
print("y_train :", y_train.shape) # (5000,)

print("x_test  :", x_test.shape)  #  (10, 100, 100, 1)
print("y_test  :", y_test.shape)  #  (10,)


# ============================================================
# Scaling 확인
# ============================================================

print()
print("Train MAX :", round(np.max(x_train), 4))
print("Train MIN :", round(np.min(x_train), 4))

print("Test MAX  :", round(np.max(x_test), 4))
print("Test MIN  :", round(np.min(x_test), 4))

#Train MAX : 1.0
#Train MIN : 0.0
#Test MAX  : 1.0
#Test MIN  : 0.0118
#exit()


# ============================================================
# 2. CNN 함수형 모델 구성 - 수평형
# ============================================================

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout


# ============================================================
# Input Layer
# ============================================================

input1 = Input(
    shape=(100, 100, 1),
    name='input1'
)


# ============================================================
# Conv Block 1
# ============================================================

conv1_out = Conv2D(
    32,
    (3, 3),
    padding='same',
    activation='relu',
    name='conv1'
)(input1)

bn1_out = BatchNormalization(
    name='bn1'
)(conv1_out)

conv2_out = Conv2D(
    32,
    (3, 3),
    padding='same',
    activation='relu',
    name='conv2'
)(bn1_out)

bn2_out = BatchNormalization(
    name='bn2'
)(conv2_out)

pool1_out = MaxPool2D(
    (2, 2),
    name='pool1'
)(bn2_out)

dropout1_out = Dropout(
    0.15,
    name='dropout1'
)(pool1_out)


# ============================================================
# Conv Block 2
# ============================================================

conv3_out = Conv2D(
    64,
    (3, 3),
    padding='same',
    activation='relu',
    name='conv3'
)(dropout1_out)

bn3_out = BatchNormalization(
    name='bn3'
)(conv3_out)

conv4_out = Conv2D(
    64,
    (3, 3),
    padding='same',
    activation='relu',
    name='conv4'
)(bn3_out)

bn4_out = BatchNormalization(
    name='bn4'
)(conv4_out)

pool2_out = MaxPool2D(
    (2, 2),
    name='pool2'
)(bn4_out)

dropout2_out = Dropout(
    0.20,
    name='dropout2'
)(pool2_out)


# ============================================================
# Conv Block 3
# ============================================================

conv5_out = Conv2D(
    128,
    (3, 3),
    padding='same',
    activation='relu',
    name='conv5'
)(dropout2_out)

bn5_out = BatchNormalization(
    name='bn5'
)(conv5_out)

conv6_out = Conv2D(
    128,
    (3, 3),
    padding='same',
    activation='relu',
    name='conv6'
)(bn5_out)

bn6_out = BatchNormalization(
    name='bn6'
)(conv6_out)

pool3_out = MaxPool2D(
    (2, 2),
    name='pool3'
)(bn6_out)

dropout3_out = Dropout(
    0.25,
    name='dropout3'
)(pool3_out)


# ============================================================
# Conv Block 4
# ============================================================

conv7_out = Conv2D(
    256,
    (3, 3),
    padding='same',
    activation='relu',
    name='conv7'
)(dropout3_out)

bn7_out = BatchNormalization(
    name='bn7'
)(conv7_out)

pool4_out = MaxPool2D(
    (2, 2),
    name='pool4'
)(bn7_out)

dropout4_out = Dropout(
    0.30,
    name='dropout4'
)(pool4_out)


# ============================================================
# Global Average Pooling
# ============================================================

gap_out = GlobalAveragePooling2D(
    name='GAP'
)(dropout4_out)


# ============================================================
# Fully Connected Layer
# ============================================================

dense1_out = Dense(
    128,
    activation='relu',
    name='dense1'
)(gap_out)

bn_dense_out = BatchNormalization(
    name='bn_dense'
)(dense1_out)

dropout5_out = Dropout(
    0.40,
    name='dropout5'
)(bn_dense_out)


# ============================================================
# Output Layer - Binary Classification
# ============================================================

output1 = Dense(
    1,
    activation='sigmoid',
    name='output1'
)(dropout5_out)


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
# 3. 컴파일
# ============================================================

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# ============================================================
# 3-1. 훈련
# ============================================================

start_time = time.time()

model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=128,
    verbose=1,
    validation_split=0.2
)

end_time = time.time()


# ============================================================
# 4. 평가
# ============================================================

print("================================ model.evaluate ================================")

test_loss, test_acc = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print('loss :', round(test_loss, 4))
print('acc  :', round(test_acc, 4))


# ============================================================
# 5. 예측
# ============================================================

y_predict_prob = model.predict(
    x_test,
    verbose=0
)


# ============================================================
# 5-1. 확률 → 0 또는 1
#     0.5 이상 → 1
#     0.5 미만 → 0
# ============================================================

y_predict = (y_predict_prob >= 0.5).astype(int).reshape(-1)


# ============================================================
# 5-2. 실제 정답 데이터 형태 정리
# ============================================================

y_test_class = np.asarray(y_test).reshape(-1).astype(int)


# ============================================================
# 5-3. Accuracy 계산
# ============================================================

acc_score = accuracy_score(
    y_test_class,
    y_predict
)

print('accuracy_score :', round(acc_score, 4))


# ============================================================
# 6. 실행 시간
# ============================================================

print(
    '걸린시간 :',
    round(end_time - start_time, 2),
    '초'
)
#####################################################################################################
#loss : 0.0
#acc  : 1.0