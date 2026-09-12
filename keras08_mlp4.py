import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x = np.array(range(10))
y = np.array([[1,2,3,4,5,6,7,8,9,10],
               [10, 9, 8, 7, 6, 5, 4, 3, 2, 1,],
               [9,8,7,6,5,4,3,2,1,0]]).transpose()

print(x.shape, y.shape)  # (10,) (10, 3)

#2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(7, input_dim=5))
model.add(Dense(5, input_dim=7))
model.add(Dense(3, input_dim=5))


#3. compile, training
model.compile(loss='mse', optimizer = 'adam')
model.fit(x, y, epochs =2000)


#4. 평가 및 추론
loss = model.evaluate(x, y)
print('손실값 :', loss)

results = model.predict(np.array([[10]])) # 학습이 끝난 model에 입력값 10을 전달하고, 모델이 계산한 출력값을 반환하라.
print('추론값 :', results) #왜 [[10]]이고 [10]이 아닌가?

# 현재 모델의 입력 형태를 보면:

# model.add(Dense(5, input_dim=1))

# 즉, 모델은 입력 특성(feature)이 1개라고 정의되어 있습니다.
# Keras의 Dense 모델에서는 일반적으로 입력 데이터를 다음 형태로 넣습니다.




