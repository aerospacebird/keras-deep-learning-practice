import numpy as np
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import time
from tensorflow.keras.layers import Dense,SimpleRNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPool2D, BatchNormalization,
    GlobalAveragePooling2D, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import os
import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. 데이터 생성
a = np.arange(1, 101).reshape(-1, 1)
print("원본 a shape:", a.shape)

size = 10

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : i + size]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print("split 후 bbb shape:", bbb.shape)

x = bbb.reshape(bbb.shape[0], 5, 2)
y = bbb[:, -1]

print("변환 후 x shape:", x.shape)
print("y shape:", y.shape)

print("\n첫 번째 샘플:")
print(x[0])
print(y[0])

print("\n마지막 샘플:")
print(x[-1])
print(y[-1])

# 2. 모델 구성
model = Sequential()
model.add(SimpleRNN(64, input_shape=(5, 2), return_sequences=True))
model.add(SimpleRNN(32))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='loss', patience=30, restore_best_weights=True, verbose=1)
rlr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=15, min_lr=1e-5, verbose=1)

model.fit(x, y,
          epochs=300,
          batch_size=8,
          callbacks=[es, rlr],
          verbose=1)

# 4. 평가
loss = model.evaluate(x, y, verbose=0)
print('loss :', loss)

# 5. 예측 (마지막 입력 샘플과 동일한 형태)
x_predict = np.array(range(91, 101)).reshape(1, 5, 2)

print("\n예측 입력 shape:", x_predict.shape)
print("예측 입력 값:")
print(x_predict)

y_predict = model.predict(x_predict, verbose=0)
print('마지막 샘플(91~100) 기반 예측 결과:', y_predict)

#####################################################################################

# Restoring model weights from the end of the best epoch: 293.
# loss : 0.0056317588314414024

# 예측 입력 shape: (1, 5, 2)
# 예측 입력 값:
# [[[ 91  92]
#   [ 93  94]
#   [ 95  96]
#   [ 97  98]
#   [ 99 100]]]
# 마지막 샘플(91~100) 기반 예측 결과: [[99.42442]]
# 이제 x_predict가 학습 때 사용한 마지막 샘플과 정확히 같은 shape (1, 5, 2)와 같은 값 구성을 가집니다.

# 예측 결과는 대략 100 근처가 나와야 정상입니다.
########################################################################################