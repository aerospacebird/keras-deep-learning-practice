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
#a = np.array(range(1,101))
#x_predict = np.array(range(96, 106)) # 101 ~106까지 찾아라?
#size = 6  # loss지표는 0.1이하, 결과는 [101,102,103,104,105,106]의 근사치가 나오면 됨.

a = np.array(range(1,101))
size= 6

print(a.shape) #(100,)
#exit()
#exit()
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i:(i+size)]
        aaa.append(subset)
    return np.array(aaa) 
bbb = split_x(a, size) 


print(bbb) 
#  [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]

bbb = split_x(a, size)
print(bbb)
# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]
#exit()
# x : 앞의 4개 열
x = bbb[:, :-1]
# y : 마지막 1개 열
y = bbb[:, -1]


print(x.shape, y.shape)  #(95, 5) (95,)

print("x :\n", x)
print("y :", y)
# x :
#  [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]
#  [ 7  8  9 10 11]
#  [ 8  9 10 11 12]
#  [ 9 10 11 12 13]
#  [10 11 12 13 14]
#  [11 12 13 14 15]
#  [12 13 14 15 16]
#  [13 14 15 16 17]
#  [14 15 16 17 18]
#  [15 16 17 18 19]
#  [16 17 18 19 20]
#  [17 18 19 20 21]
#  [18 19 20 21 22]
#  [19 20 21 22 23]
#  [20 21 22 23 24]
#  [21 22 23 24 25]
#  [22 23 24 25 26]
#  [23 24 25 26 27]
#  [24 25 26 27 28]
#  [25 26 27 28 29]
#  [26 27 28 29 30]
#  [27 28 29 30 31]
#  [28 29 30 31 32]
#  [29 30 31 32 33]
#  [30 31 32 33 34]
#  [31 32 33 34 35]
#  [32 33 34 35 36]
#  [33 34 35 36 37]
#  [34 35 36 37 38]
#  [35 36 37 38 39]
#  [36 37 38 39 40]
#  [37 38 39 40 41]
#  [38 39 40 41 42]
#  [39 40 41 42 43]
#  [40 41 42 43 44]
#  [41 42 43 44 45]
#  [42 43 44 45 46]
#  [43 44 45 46 47]
#  [44 45 46 47 48]
#  [45 46 47 48 49]
#  [46 47 48 49 50]
#  [47 48 49 50 51]
#  [48 49 50 51 52]
#  [49 50 51 52 53]
#  [50 51 52 53 54]
#  [51 52 53 54 55]
#  [52 53 54 55 56]
#  [53 54 55 56 57]
#  [54 55 56 57 58]
#  [55 56 57 58 59]
#  [56 57 58 59 60]
#  [57 58 59 60 61]
#  [58 59 60 61 62]
#  [59 60 61 62 63]
#  [60 61 62 63 64]
#  [61 62 63 64 65]
#  [62 63 64 65 66]
#  [63 64 65 66 67]
#  [64 65 66 67 68]
#  [65 66 67 68 69]
#  [66 67 68 69 70]
#  [67 68 69 70 71]
#  [68 69 70 71 72]
#  [69 70 71 72 73]
#  [70 71 72 73 74]
#  [71 72 73 74 75]
#  [72 73 74 75 76]
#  [73 74 75 76 77]
#  [74 75 76 77 78]
#  [75 76 77 78 79]
#  [76 77 78 79 80]
#  [77 78 79 80 81]
#  [78 79 80 81 82]
#  [79 80 81 82 83]
#  [80 81 82 83 84]
#  [81 82 83 84 85]
#  [82 83 84 85 86]
#  [83 84 85 86 87]
#  [84 85 86 87 88]
#  [85 86 87 88 89]
#  [86 87 88 89 90]
#  [87 88 89 90 91]
#  [88 89 90 91 92]
#  [89 90 91 92 93]
#  [90 91 92 93 94]
#  [91 92 93 94 95]
#  [92 93 94 95 96]
#  [93 94 95 96 97]
#  [94 95 96 97 98]
#  [95 96 97 98 99]]

# y : [  6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23
#   24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40  41
#   42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59
#   60  61  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77
#   78  79  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95
#   96  97  98  99 100]

#exit()
# RNN 입력 형태로 reshape (samples, timesteps, features)
x = x.reshape(x.shape[0], x.shape[1], 1)   # RNN 입력 x.shape : (95, 5, 1)
print("RNN 입력 x.shape :", x.shape)
#exit()

# 2. 모델 구성
model = Sequential()
model.add(SimpleRNN(95, input_shape=(5, 1), return_sequences=True))  # timesteps=4, features=1
model.add(SimpleRNN(10))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True, verbose=1)
rlr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=1e-5, verbose=1)

model.fit(x, y,
          epochs=500,
          batch_size=1,
          callbacks=[es, rlr],
          verbose=1)

# 4. 평가, 예측
result = model.evaluate(x, y, verbose=0)
print('loss :', result)

# 예측 (timesteps=5에 맞춰서 입력)
x_predict = np.array(range(96, 106)).reshape(-1, 5, 1)
y_predict = model.predict(x_predict, verbose=0)
print('[101,102,103,104,105,106]의 예측 결과:', y_predict)

model.summary()
#########################################################################################
# Restoring model weights from the end of the best epoch: 461.
# loss : 0.0021549479570239782
# [101,102,103,104,105,106]의 예측 결과: [[100.88314]
#  [103.68811]]
#########################################################################################
# Restoring model weights from the end of the best epoch: 369.
# loss : 0.0016575799090787768
# [101,102,103,104,105,106]의 예측 결과: [[100.91672 ]
#  [103.683205]]
#########################################################################################