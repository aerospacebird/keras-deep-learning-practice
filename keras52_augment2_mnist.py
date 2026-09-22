# 50-2 카피 (최종 수정본)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv2D, Dropout, Dense,
    BatchNormalization, MaxPool2D, GlobalAveragePooling2D
)
from sklearn.metrics import accuracy_score
import time
import pandas as pd
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense,Dropout, Flatten
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense
)
import time
from sklearn.metrics import accuracy_score
# 1. 데이터 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()

################# 데이터 증강 ####################
datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    width_shift_range=0.1,
    rotation_range=15,
    fill_mode='nearest'
)

augment_size = 40000  
randidx = np.random.randint(x_train.shape[0], size=augment_size)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

# 채널 차원 추가
x_augmented = x_augmented.reshape(x_augmented.shape[0], 28, 28, 1)

# 증강 수행
x_augmented = datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    shuffle=False
).next()[0]

print("증강 후 shape:", x_augmented.shape)  # (40000, 28, 28, 1)

# 원본 데이터도 채널 차원 추가 + 스케일링
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255.
x_test  = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255.

# 원본 + 증강 데이터 합치기
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print("최종 train shape:", x_train.shape, y_train.shape)  # (100000, 28, 28, 1) (100000,)
print(np.unique(y_train, return_counts=True))
#exit()

# # ============================================================
# # 2. Model Construction - DNN
# # ============================================================

# model = Sequential()

# # 28×28×1 image → 784-dimensional vector
# model.add(Flatten(input_shape=(28, 28, 1)))

# # Fully Connected Layer
# model.add(Dense(128, activation='relu'))
# model.add(BatchNormalization())
# # Dropout for regularization
# model.add(Dropout(0.2))

# # Fully Connected Layer
# model.add(Dense(64, activation='relu'))
# model.add(BatchNormalization())
# # Dropout for regularization
# model.add(Dropout(0.2))

# # Fully Connected Layer
# model.add(Dense(32, activation='relu'))
# model.add(BatchNormalization())
# # Dropout for regularization
# model.add(Dropout(0.2))

# # Fully Connected Layer
# model.add(Dense(16, activation='relu'))
# model.add(BatchNormalization())
# # Output Layer - 10 classes
# model.add(Dense(10, activation='softmax'))

# # Model Summary
# model.summary()

# ============================================================
# 모델 구성 (Functional API)
# ============================================================

input1 = Input(shape=(28, 28, 1), name='input1')

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

# Global Average Pooling
x = GlobalAveragePooling2D(name='GAP')(x)

# Fully Connected
x = Dense(128, activation='relu', name='dense1')(x)
x = BatchNormalization(name='bn_dense')(x)
x = Dropout(0.40, name='dropout5')(x)

# ★ 출력층 (10클래스 + softmax)
output1 = Dense(10, activation='softmax', name='output1')(x)

model = Model(inputs=input1, outputs=output1)
model.summary()


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
    epochs=100,                    # 100은 너무 김 → 50으로 권장 (필요시 늘리세요)
    batch_size=5000,
    verbose=1,
    validation_split=0.2
)

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
y_predict = np.argmax(y_predict_prob, axis=1)          # ★ 10클래스용

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score :', round(acc_score, 4))
print('걸린시간 :', round(end_time - start_time, 2), '초')

################################################################################################################
# =============================== model.evaluate ===============================
# 313/313 [==============================] - 9s 30ms/step - loss: 0.2952 - acc: 0.9127
# loss: 0.2952014207839966
# acc : 0.9126999974250793
# 313/313 [==============================] - 7s 20ms/step
# accuracy_score : 0.9127
# 걸린시간 : 2752.47 초

###############################################################################################################
# ================================ model.evaluate ================================
# 313/313 [==============================] - 1s 2ms/step - loss: 0.2592 - accuracy: 0.9387
# loss : 0.2592
# acc  : 0.9387
# accuracy_score : 0.9387
# 걸린시간 : 4106.78 초

################################################################################################################
# Epoch 346/500
# 12/12 [==============================] - ETA: 0s - loss: 0.0105 - accuracy: 0.9964


#################################################################################################################
# ================================ model.evaluate ================================
# 313/313 [==============================] - 1s 2ms/step - loss: 0.3900 - accuracy: 0.9391
# loss : 0.39
# acc  : 0.9391
# accuracy_score : 0.9391
# 걸린시간 : 1981.05 초
################################################################################################################
================================ model.evaluate ================================
# 313/313 [==============================] - 6s 11ms/step - loss: 0.0196 - accuracy: 0.9954
# loss : 0.0196
# acc  : 0.9954
# accuracy_score : 0.9954
# 걸린시간 : 1981.17 초