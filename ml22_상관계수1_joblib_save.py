
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
#fetch_california_housing(return_X_y=True)
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from xgboost import XGBRegressor
#1. 데이터

datasets = load_iris()
print(datasets.feature_names)

x = datasets['data']
y = datasets.target


df = pd.DataFrame(x, columns=datasets.feature_names)
print(df)

df['target'] = y
print(df)
#[150 rows x 5 columns]

exit()

# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
#      sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# 0                  5.1               3.5                1.4               0.2
# 1                  4.9               3.0                1.4               0.2
# 2                  4.7               3.2                1.3               0.2
# 3                  4.6               3.1                1.5               0.2
# 4                  5.0               3.6                1.4               0.2
# ..                 ...               ...                ...               ...
# 145                6.7               3.0                5.2               2.3
# 146                6.3               2.5                5.0               1.9
# 147                6.5               3.0                5.2               2.0
# 148                6.2               3.4                5.4               2.3
# 149                5.9               3.0                5.1               1.8

# [150 rows x 4 columns]

# joblib, pickle로 save하기

#1. 데이터

#x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=444,
    #stratify=y,
)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


parameters = {'n_estimator': 2000,
              'learning_rate': 0.5,
              'max_depth': 7,
              'gamma': 1,
              'min_chind_weight': 1,
              'subsample': 1,
              'colsample_bytree': 1,
              'colsample_bylevel': 1,
              'colsample_bynode': 1,
              'reg_alpha': 0,
              'reg_lambda': 1,
              'random_state': 842,
              #'verbose' :0,              
              }

#2. 모델
#model = DecisionTreeRegressor()
#model = RandomForestRegressor()
model = XGBRegressor(    
    #**parameters,  #  **args: 딕셔너리로 받는다. *args: 튜플로 받는다.  
    )

model.set_params(**parameters,
                 early_stopping_rounds=20,
                 )




# model = BaggingRegressor(
#     DecisionTreeRegressor(),
#     n_estimators=10,  # 일종의 epochs입니다.
#     random_state=42,
#     n_jobs=-1,
#     max_samples=0.8,
#     #bootstrap=True, # 디폴트 입니다. # 데이터의 중복을 허용한다.
#     bootstrap=False # 데이터의 중복 허용 하지 않음.
# )

#3. 훈련
model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)],
          verbose = 1, 
          )

#4. 평가, 예측

result = model.score(x_test, y_test)
print('최종 점수:', result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred )
print('r2:', r2)



# path = './_save/joblib/'
# import joblib

# joblib.dump(model, path + 'ml22_joblib2_save.joblib')

path = './_save/pickle/'
import joblib
import pickle

#joblib.dump(model, path + 'ml19_joblib1_save.joblib')
pickle.dump(model, open(path + 'ml22_pickle2_save.pickle', 'wb'))

#####################################    joblibd으로 저장하기     ###########################################
# 최종 점수: 0.8342829942703247
# r2: 0.8342829942703247
#####################################    pickle로 저장하기        ##########################################
# 최종 점수: 0.8342829942703247
# r2: 0.8342829942703247
###########################################################################################################