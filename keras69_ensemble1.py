import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T          # (100, 2) 삼성 종가, 하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511),
                        range(150, 250)]).transpose()            # (100, 3) 원유가, 환율, 금시세
y = np.array(range(3001, 3101))                                  # (100,)   화성 화씨 온도

# train / test 분리
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1_datasets, x2_datasets, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("x1_train:", x1_train.shape)   # (80, 2)
print("x1_test :", x1_test.shape)    # (20, 2)
print("x2_train:", x2_train.shape)   # (80, 3)
print("x2_test :", x2_test.shape)    # (20, 3)
print("y_train :", y_train.shape)    # (80,)
print("y_test  :", y_test.shape)     # (20,)

#2-1 모델1 (x1)
input1 = Input(shape=(2,))
dense1 = Dense(10, activation="relu", name="han1")(input1)
dense2 = Dense(20, activation="relu", name="han2")(dense1)
dense3 = Dense(30, activation="relu", name="han3")(dense2)
dense4 = Dense(40, activation="relu", name="han4")(dense3)
output1 = Dense(5, activation="relu", name="han5")(dense4)

#2-2 모델2 (x2)
input20 = Input(shape=(3,))
dense21 = Dense(50, activation="relu", name="han21")(input20)
dense22 = Dense(40, activation="relu", name="han22")(dense21)
dense23 = Dense(30, activation="relu", name="han23")(dense22)
dense24 = Dense(20, activation="relu", name="han24")(dense23)
output21 = Dense(3, activation="relu", name="han25")(dense24)

#2-3 모델 합치기
merge1 = Concatenate(name='mg1')([output1, output21])
merge2 = Dense(10, name='mg2')(merge1)
merge3 = Dense(5, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input20], outputs=last_output)
model.summary()

#3. 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    patience=20,
    mode='min',
    restore_best_weights=True,
    verbose=1
)

model.fit(
    [x1_train, x2_train], y_train,
    validation_data=([x1_test, x2_test], y_test),
    epochs=100,
    batch_size=8,
    callbacks=[es],
    verbose=1
)

#4. 평가
result = model.evaluate([x1_test, x2_test], y_test)
print('loss :', result)

#5. 예측
x1_pred = np.array([range(100, 106), range(400, 406)]).T          # (6, 2)
x2_pred = np.array([range(200, 206), range(510, 516),
                    range(250, 256)]).T                            # (6, 3)

y_pred = model.predict([x1_pred, x2_pred])
print('예측값 :\n', y_pred)

####################################################################################################
# loss : 112.11324310302734
# 1/1 [==============================] - 0s 145ms/step
# 예측값 :
#  [[3107.7058]
#  [3110.564 ]
#  [3110.564 ]
#  [3113.0024]
#  [3113.0024]
#  [3115.1765]]