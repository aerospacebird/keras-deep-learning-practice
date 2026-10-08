
# 69-1 copy (수정 완료)
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T                    # (100, 2)
x2_datasets = np.array([range(101, 201), range(411, 511),
                        range(150, 250)]).transpose()                       # (100, 3)
x3_datasets = np.array([range(100), range(301, 401),
                        range(77, 177), range(33, 133)]).T                  # (100, 4)

y1 = np.array(range(3001, 3101))      # (100,)
y2 = np.array(range(13001, 13101))    # (100,)  # 비트코인 가격

# train / test 분리 (y1, y2 둘 다)
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y1, y2,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("x1_train:", x1_train.shape)   # (80, 2)
print("x1_test :", x1_test.shape)    # (20, 2)
print("x2_train:", x2_train.shape)   # (80, 3)
print("x2_test :", x2_test.shape)    # (20, 3)
print("x3_train:", x3_train.shape)   # (80, 4)
print("x3_test :", x3_test.shape)    # (20, 4)
print("y1_train:", y1_train.shape)   # (80,)
print("y1_test :", y1_test.shape)    # (20,)
print("y2_train:", y2_train.shape)   # (80,)
print("y2_test :", y2_test.shape)    # (20,)

#2-1 모델1 (x1)
input1 = Input(shape=(2,), name='input1')
dense1 = Dense(10, activation="relu", name="han1")(input1)
dense2 = Dense(20, activation="relu", name="han2")(dense1)
dense3 = Dense(30, activation="relu", name="han3")(dense2)
dense4 = Dense(40, activation="relu", name="han4")(dense3)
output1 = Dense(5, activation="relu", name="han5")(dense4)

#2-2 모델2 (x2)
input2 = Input(shape=(3,), name='input2')
dense21 = Dense(50, activation="relu", name="han21")(input2)
dense22 = Dense(40, activation="relu", name="han22")(dense21)
dense23 = Dense(30, activation="relu", name="han23")(dense22)
dense24 = Dense(20, activation="relu", name="han24")(dense23)
output2 = Dense(3, activation="relu", name="han25")(dense24)

#2-3 모델3 (x3)
input3 = Input(shape=(4,), name='input3')
dense31 = Dense(50, activation="relu", name="han31")(input3)
dense32 = Dense(40, activation="relu", name="han32")(dense31)
dense33 = Dense(30, activation="relu", name="han33")(dense32)
dense34 = Dense(20, activation="relu", name="han34")(dense33)
output3 = Dense(3, activation="relu", name="han35")(dense34)

#2-4 모델 합치기
merge1 = Concatenate(name='mg1')([output1, output2, output3])
merge2 = Dense(10, activation='relu', name='mg2')(merge1)
merge3 = Dense(5, activation='relu', name='mg3')(merge2)
merge4 = Dense(5, activation='relu', name='mg4')(merge3)

#2-5 분기 (multi-output)
last_output1 = Dense(1, name='y1_output')(merge4)   # y1 예측
last_output2 = Dense(1, name='y2_output')(merge4)   # y2 예측

model = Model(inputs=[input1, input2, input3], outputs=[last_output1, last_output2])
model.summary()

#3. 컴파일 및 훈련
model.compile(
    loss={'y1_output': 'mse', 'y2_output': 'mse'},
    optimizer='adam',
    metrics={'y1_output': 'mse', 'y2_output': 'mse'}
)

es = EarlyStopping(
    monitor='val_loss',
    patience=20,
    mode='min',
    restore_best_weights=True,
    verbose=1
)

model.fit(
    [x1_train, x2_train, x3_train],
    {'y1_output': y1_train, 'y2_output': y2_train},
    validation_data=(
        [x1_test, x2_test, x3_test],
        {'y1_output': y1_test, 'y2_output': y2_test}
    ),
    epochs=100,
    batch_size=8,
    callbacks=[es],
    verbose=1
)

#4. 평가
result = model.evaluate(
    [x1_test, x2_test, x3_test],
    {'y1_output': y1_test, 'y2_output': y2_test},
    verbose=0
)
#리스트([ ])와 중괄호({ })를 함께 사용하는 의미는 Keras의 Multi-Input / Multi-Output 모델에서 입력과 출력을 구분해서 전달.
print('loss (total, y1, y2):', result)

#5. 예측
x1_pred = np.array([range(100, 106), range(400, 406)]).T                    # (6, 2)
x2_pred = np.array([range(200, 206), range(510, 516),
                    range(250, 256)]).T                                      # (6, 3)
x3_pred = np.array([range(100, 106), range(400, 406),
                    range(177, 183), range(133, 139)]).T                     # (6, 4)

y_pred1, y_pred2 = model.predict([x1_pred, x2_pred, x3_pred], verbose=0)

print('y_pred1 (y1 예측값):\n', y_pred1)
print('y_pred2 (y2 예측값):\n', y_pred2)
# import numpy as np
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Input, Concatenate
# from tensorflow.keras.callbacks import EarlyStopping

# #1. 데이터
# x1_datasets = np.array([range(100), range(301, 401)]).T                    # (100, 2)
# x2_datasets = np.array([range(101, 201), range(411, 511),
#                         range(150, 250)]).transpose()                       # (100, 3)
# x3_datasets = np.array([range(100), range(301, 401),
#                         range(77, 177), range(33, 133)]).T                  # (100, 4)
# y1 = np.array(range(3001, 3101)) 
# y2 = np.array(range(13001, 13101)) #비트코인 가격
#                                    # (100,)

# # train / test 분리
# x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y1_train, y2_test = train_test_split(
#     x1_datasets, x2_datasets, x3_datasets, y1,y2
#     test_size=0.2,
#     random_state=42,
#     shuffle=True
# )

# print("x1_train:", x1_train.shape)   # (80, 2)
# print("x1_test :", x1_test.shape)    # (20, 2)
# print("x2_train:", x2_train.shape)   # (80, 3)
# print("x2_test :", x2_test.shape)    # (20, 3)
# print("x3_train:", x3_train.shape)   # (80, 4)
# print("x3_test :", x3_test.shape)    # (20, 4)
# print("y_train :", y_train.shape)    # (80,)
# print("y_test  :", y_test.shape)     # (20,)

# #2-1 모델1 (x1)
# input1 = Input(shape=(2,))
# dense1 = Dense(10, activation="relu", name="han1")(input1)
# dense2 = Dense(20, activation="relu", name="han2")(dense1)
# dense3 = Dense(30, activation="relu", name="han3")(dense2)
# dense4 = Dense(40, activation="relu", name="han4")(dense3)
# output1 = Dense(5, activation="relu", name="han5")(dense4)

# #2-2 모델2 (x2)
# input20 = Input(shape=(3,))
# dense21 = Dense(50, activation="relu", name="han21")(input20)
# dense22 = Dense(40, activation="relu", name="han22")(dense21)
# dense23 = Dense(30, activation="relu", name="han23")(dense22)
# dense24 = Dense(20, activation="relu", name="han24")(dense23)
# output21 = Dense(3, activation="relu", name="han25")(dense24)

# #2-3 모델3 (x3)  ★ shape=(4,)로 수정
# input30 = Input(shape=(4,))
# dense31 = Dense(50, activation="relu", name="han31")(input30)
# dense32 = Dense(40, activation="relu", name="han32")(dense31)
# dense33 = Dense(30, activation="relu", name="han33")(dense32)
# dense34 = Dense(20, activation="relu", name="han34")(dense33)
# output31 = Dense(3, activation="relu", name="han35")(dense34)

# #2-4 모델 합치기
# merge1 = Concatenate(name='mg1')([output1, output21, output31])
# merge2 = Dense(10, name='mg2')(merge1)
# merge3 = Dense(5, name='mg3')(merge2)
# merge4 = Dense(5, name='mg4')(merge3)

# last_dense1 = Dense(10, name='ld1')((merge3))
# last_dense2 = Dense(10, name='ld2')((last_dense1))
# last_output1 = Dense(10, name='last1')((merge3))

# #2-6. 분기
# last_output2 = Dense(1, name='last2')(merge3)
# model=Model(input=[input1, input21, input31], outputs=[last_output1, last_output2])

# last_output1 = Dense(1, name='last')(merge4)

# last_output1 = Dense(1, name='last')(merge4)

# model = Model(inputs=[input1, input20, input30], outputs=last_output)
# model.summary()

# #3. 컴파일 및 훈련
# model.compile(loss='mse', optimizer='adam')

# es = EarlyStopping(
#     monitor='val_loss',
#     patience=20,
#     mode='min',
#     restore_best_weights=True,
#     verbose=1
# )

# model.fit(
#     [x1_train, x2_train, x3_train], y_train,
#     validation_data=([x1_test, x2_test, x3_test], y_test),
#     epochs=100,
#     batch_size=8,
#     callbacks=[es],
#     verbose=1
# )

# #4. 평가
# result = model.evaluate([x1_test, x2_test, x3_test], y1_test, y2_test)
# print('loss :', result)

# #5. 예측
# x1_pred = np.array([range(100, 106), range(400, 406)]).T                    # (6, 2)
# x2_pred = np.array([range(200, 206), range(510, 516),
#                     range(250, 256)]).T                                      # (6, 3)
# x3_pred = np.array([range(100, 106), range(400, 406),
#                     range(177, 183), range(133, 139)]).T                     # (6, 4)

# y_pred1, y_pred2 = model.predict([x1_pred, x2_pred, x3_pred])

# print('예측값 :\n', y_pred1, y_pred2)

#############################################################################################################
# loss : 14.175329208374023
# 1/1 [==============================] - 0s 96ms/step
# 예측값 :
#  [[3107.1306]
#  [3111.9636]
#  [3120.6638]
#  [3127.429 ]
#  [3135.5774]
#  [3140.057 ]]
#############################################################################################################
# loss (total, y1, y2): [252042.875, 234998.046875, 17044.828125, 234998.046875, 17044.828125]
# y_pred1 (y1 예측값):
#  [[3547.8848]
#  [3552.8027]
#  [3555.1567]
#  [3559.8762]
#  [3564.474 ]
#  [3571.4265]]
# y_pred2 (y2 예측값):
#  [[12980.975]
#  [12998.966]
#  [13007.577]
#  [13024.842]
#  [13041.661]
#  [13067.094]]
###########################################################################################################