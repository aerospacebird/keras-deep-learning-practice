import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# x_train = np.array([1,2,3,4,5,6,7])
# y_train = np.array([1,2,3,4,5,6,7])


# x_test = np.array([8,9,10])
# y_test = np.array([8,9,10])

#[실습 찾아보기 넘파이 리스트의 슬라이싱========> 7:3 으로 나누자]

x_train = x[:7]     # 0번 ~ 6번 → 7개    or  x_train = x[0:7] x_train: [1 2 3 4 5 6 7]
y_train = y[:7]     # y_train: [1 2 3 4 5 6 7]

x_test = x[7:]      # 7번 ~ 끝 → 3개    x_test : [ 8  9 10]
y_test = y[7:]      #                  y_test : [ 8  9 10]

print("x_train:", x_train)
print("y_train:", y_train)

print("x_test :", x_test)
print("y_test :", y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 