from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import time

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (309, 10) (133, 10) (309,) (133,)

# 2. 모델

def build_model(drop=0.5, optimizer='adam', activation='relu', node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.001):
    inputs = Input(shape=(10,), name='inputs')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation, name='hidden4')(x)
    x = Dense(node5, activation=activation, name='hidden5')(x)
    outputs = Dense(1, activation='linear', name='outputs')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=optimizer, metrics=['mae'], loss='mse')
    return model

def create_hyperparameter():
    batchs = [32, 16, 8, 1, 64]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16]
    node4 = [128, 64, 32, 16]
    node5 = [128, 64, 32, 16]
    return {
        'batch_size': batchs,
        'optimizer': optimizers,
        'drop': dropouts,
        'activation': activations,
        'node1': node1,
        'node2': node2,
        'node3': node3,
        'node4': node4,
        'node5': node5,
    }
hyperparameters = create_hyperparameter()
print(hyperparameters)

from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

keras_model = KerasRegressor(build_fn=build_model, verbose=1)


model = RandomizedSearchCV(
    keras_model, 
    hyperparameters, 
    cv=3,
    random_state=333,
    n_iter=10,
    verbose=1,
    refit=True,
    # n_jobs=4,
)

# 3. 훈련

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    restore_best_weights=True,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=8,
    verbose=1,
    factor=0.8,
)


start = time.perf_counter()
model.fit(x_train, y_train, epochs=100, verbose=1, callbacks=[es, rlr], validation_split=0.1)
end = time.perf_counter()

print("===============================================")
# best_estimator_: 최적 파라미터가 적용되고 재학습까지 완료된 모델 객체
print("최적의 매개변수 :", model.best_estimator_)
print("===============================================")
# best_params_: 교차검증에서 가장 높은 평균 점수를 기록한 하이퍼파라미터 조합
print("최적의 파라미터 :", model.best_params_)
print("===============================================")



# 4. 평가, 예측
print("model.score :", model.score(x_test, y_test))

print("best_score :", model.best_score_)

print("걸린시간 :", round(end - start, 2), "초")

####################################################################################################################
# ===============================================
# 최적의 매개변수 : <keras.wrappers.scikit_learn.KerasRegressor object at 0x000002608640AC20>
# ===============================================
# 최적의 파라미터 : {'optimizer': 'rmsprop', 'node5': 32, 'node4': 64, 'node3': 32, 'node2': 16, 'node1': 16, 'drop': 0.3, 'batch_size': 8, 'activation': 'selu'}
# ===============================================
# 12/12 [==============================] - 0s 2ms/step - loss: 3107.1565 - mae: 46.3189
# model.score : -3107.156494140625
# best_score : -3134.998779296875
# 걸린시간 : 214.22 초

##################################################################################################################