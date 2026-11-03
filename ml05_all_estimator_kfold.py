
# classifier cross validation ~

import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    cross_val_predict
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    KFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from sklearn.utils import all_estimators
import sklearn as sk
import warnings
warnings.filterwarnings('ignore')
# ==================================================
# 1. Dataset
# ==================================================

# Boston Housing 대신 California Housing 사용
dataset = fetch_california_housing()

x = dataset.data
y = dataset.target

print("X shape:", x.shape)
print("Y shape:", y.shape)

# ==================================================
# 2. Train / Test Split
# ==================================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    shuffle=True,
    random_state=333
)

print("Train:", x_train.shape)
print("Test :", x_test.shape)

# ==================================================
# 3. Scaling
# ==================================================

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ==================================================
# 4. K-Fold
# ==================================================

n_split = 5

kfold = KFold(
    n_splits=n_split,
    shuffle=True,
    random_state=333
)

# ==================================================
# 5. Model
# ==================================================
# n_split = 5
# kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

# * 2. 모델
# model = DecisionTreeClassifier()
all = all_estimators(type_filter= "classifier")
#all = all_estimators(type_filter= "regressor")

print('all models :', all)
print('sk version :', sk.__version__)
print('모델의 개수 :', len(all))


for name, model in all:
    #print(name)
  try:
    model = model()



    #3. 훈련
    model.fit(x_train, y_train)

    #4. 평가
    acc= model.score(x_test, y_test)
    print(name, "의 정답율:", acc)
  except:
    print(name, '모델 오류')
# # ==================================================
# # 6. Cross Validation
# # ==================================================

# scores = cross_val_score(
#     model,
#     x_train,
#     y_train,
#     cv=kfold,
#     scoring="r2"
# )

# print("\nK-Fold R2 Scores:", scores)
# print("Mean R2:", np.mean(scores))

# # ==================================================
# # 7. Cross Validation Prediction
# # ==================================================

# y_cv_predict = cross_val_predict(
#     model,
#     x_train,
#     y_train,
#     cv=kfold
# )

# print("\nCross Validation Results")

# print("MAE:",
#       mean_absolute_error(y_train, y_cv_predict))

# print("RMSE:",
#       np.sqrt(mean_squared_error(y_train, y_cv_predict)))

# print("R2:",
#       r2_score(y_train, y_cv_predict))

# # ==================================================
# # 8. Final Training
# # ==================================================

# model.fit(x_train, y_train)

# # ==================================================
# # 9. Test Evaluation
# # ==================================================

# y_predict = model.predict(x_test)

# print("\nFinal Test Results")

# print("MAE:",
#       mean_absolute_error(y_test, y_predict))

# print("RMSE:",
#       np.sqrt(mean_squared_error(y_test, y_predict)))

# print("R2:",
#       r2_score(y_test, y_predict))


##########################################################################################################################################
# acc : [0.95833333 0.91666667 0.95833333 1.         0.95833333] 
# 평균 acc : 0.96



# from sklearn.datasets import fetch_california_housing, load_diabetes
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense, Input
# from tensorflow.keras.datasets import boston_housing
# from sklearn.model_selection import train_test_split
# import numpy as np
# import time
# import numpy as np
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split, KFold, cross_val_score
# from sklearn.tree import DecisionTreeClassifier
# import numpy as np
# import pandas as pd
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from sklearn.model_selection import train_test_split
# import time
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.datasets import load_breast_cancer
# from sklearn.metrics import accuracy_score #

# #1. 데이터
# #datasets = load_diabetes()
# #x = datasets.data
# #y = datasets.target

# (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
# print(x_train.shape, x_test.shape) #(404, 13) (102, 13)
# print(y_train.shape, y_test.shape) #(404,) (102,)

# # ==================================================
# # Train / Validation 분리
# # ==================================================
# # 현재 train 데이터의 25%를 validation으로 사용

# from sklearn.preprocessing import MinMaxScaler,StandardScaler, MaxAbsScaler, RobustScaler
# #scaler = MinMaxScaler()
# #scaler = StandardScaler()
# #scaler = MaxAbsScaler()
# scaler = RobustScaler()

# #scaler.fit(x_train)  #준비
# #x_train = scaler.transform(x_train)  #변환

# x_train = scaler.fit_transform(x_train)  #변환
# x_test = scaler.transform(x_test)

# print(np.min(x_train), np.max(x_train))
# print(np.min(x_test), np.max(x_test))
# #0.0 1.0000000000000002
# #-0.0019120458891013214 1.1478180091225068
# #exit()

# x_train, x_val, y_train, y_val = train_test_split(
#     x_train,
#     y_train,
#     train_size=0.75,
#     shuffle=True,
#     random_state=500
# )
# # Validation을 만들려면 **두 번째 train_test_split()**을 사용해야 합니다.

# #x_train, x_test, y_train, y_test = train_test_split(         데이터 세트가 이미 나눠져 있어서 불필요하다.
#     #x, y,
# #    random_state=42,
# #)



# #1. 데이터
# path = './_data/kaggle_santander/'  # 상대경로

# #path = "c:/study/_data/kaggle_santander/"  # 절대경로
# #path = "c:/study/_data/ddarung/"  # 슬래시 역슬래시 상관없어
# #path = "c://study//_data//ddarung/"  # 슬래시 2개도 가능하다.  가독성있게 작성하라.

# train_csv = pd.read_csv(path + "train.csv", index_col=0) #1459 rows x 11columns ------->>>>#1459 rows x 10 columns index_col=0 때문이다.
# print(train_csv.shape) #[200000 rows x 200 columns]

# test_csv = pd.read_csv(path + "test.csv", index_col=0)
# print(test_csv) #[200000 rows x 200 columns]

# submission = pd.read_csv(path + "sample_submission.csv", index_col=0)
# print(submission) #[6493 rows x 1 columns]

# print(train_csv.shape) #[200000 rows x 200 columns]
# print(test_csv.shape) # [200000 rows x 200 columns]
# print(submission.shape) # [200000 rows x 200 columns]



# print(train_csv.columns) 

# print(train_csv.info())
# print(test_csv.info())

# print(train_csv.describe()) 

# ################################### 결측치 확인 ###########################################################################
# print(train_csv.isna().sum())
# print(test_csv.isnull().sum())

# ###################################### x, y 분리 ###########################################################################

# x = train_csv.drop(['target'], axis=1)
# print(x)   #

# y = train_csv['target']
# print(y)    
# print(x.shape, y.shape)    #(200000, 200) (200000,)

# print(np.unique(y, return_counts=True))
# #(array([0, 1]), array([179902, 20098]))


# """
# x_train, x_test, y_train, y_test = train_test_split(
#                                    x, y,
    
#                                    random_state=42,
# )
# """
# ####################################### submit 물밑작업 #######################################################

# print(test_csv.info())

# ###################################결측치 처리 2.  평균값 넣기     #################################################
# #test_csv = test_csv.fillna(test_csv.mean())
# #print(test_csv.info())  # (6493, 8)
# #print(test_csv.shape)  # (6493, 8)
# x_train, x_test, y_train, y_test = train_test_split(x, y,
#                  train_size=0.75,
#                  #test_size=0.25,
#                  shuffle=True,   # 디폴트 섞는다. 75%/25% 비율로 섞는다.
#                  random_state=337, # # 42  디폴트 값이다. 이값을 변경해도 성능에 영향을 줄수가 있다.
#                  stratify=y #훈련 데이터와 테스트 데이터를 나눌 때, y의 클래스 비율이 원본 데이터와 최대한 동일하게 유지되도록 분할하는 옵션입니다.               
# )
# print(x_train.shape, x_test.shape) # (398, 30)  (171, 30)
# print(y_train.shape, y_test.shape) # (398, )    (171, )
# #############################################################################################
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# #scaler = MinMaxScaler()
# #scaler = StandardScaler()
# #scaler = MaxAbsScaler()
# scaler = RobustScaler()

# #scaler.fit(x_train)  #준비
# #x_train = scaler.transform(x_train)  #변환

# x_train = scaler.fit_transform(x_train)  #변환
# x_test = scaler.transform(x_test)

# print(np.min(x_train), np.max(x_train))
# print(np.min(x_test), np.max(x_test))
# #0.0 1.0000000000000002
# #-0.07306255183145582 1.0796097666145639
# #exit()
# #print(np.unique(y_train, return_counts=True))
# #print(np.unique(y_test, return_counts=True))
# #1. data

# # x, y = load_iris(return_X_y=True)
# # print(x, y)

# n_split = 5
# kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

# #2. model

# model = DecisionTreeClassifier()


# #3. Train
# scores = cross_val_score(model, x, y, cv=kfold)

# print('acc :', scores, '\n평균 acc : ', round(np.mean(scores),2))

###################################################################################################################################

# acc : [0.83705 0.83505 0.8347  0.8345  0.8359 ] 
# 평균 acc :  0.84

####################################################################################################################################

# K-Fold R2 Scores: [0.59002027 0.60337837 0.65782417 0.60242728 0.59933385]
# Mean R2: 0.6105967865615823

# Cross Validation Results
# MAE: 0.46237645591085275
# RMSE: 0.7232739534790072
# R2: 0.6112521184245099

# Final Test Results
# MAE: 0.4561556855620155
# RMSE: 0.7170201667352283
# R2: 0.5964784745401488
# PS C:\study> 
#####################################################################################################################################