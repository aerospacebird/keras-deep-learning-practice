import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

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

# from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                 x, y,
                 train_size=0.75,
                 #test_size=0.3,
                 shuffle=True,   # 디폴트 섞는다. 75%/25% 비율로 섞는다.
                 random_state=500,   # 42  디폴트 값이다. 이값을 변경해도 성능에 영향을 줄수가 있다.
)



# X: 데이터 특징(Features), y: 타깃/라벨(Labels)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


print("x_train:", x_train)
print("x_test :", x_test)
#print("y_train:", y_train)

#print("x_test :", x_test)
print("y_train:", y_train)
print("y_test :", y_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(7, input_dim=1))
model.add(Dense(5))
model.add(Dense(1))

#3. compile & training
model.compile(loss='mse', optimizer= 'adam')
model.fit(x_train, y_train, epochs=3000)

#4. 평가및 추론
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

results = model.predict(np.array([11]))
print('추론값은 :', results)  # 추론값은 : [[9.535717]]