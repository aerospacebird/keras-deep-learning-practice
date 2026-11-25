
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
#fetch_california_housing(return_X_y=True)
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from xgboost import XGBRegressor
import joblib


#1. 데이터

x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=444,
    #stratify=y,
)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# parameters = {'n_estimator': 2000,
#               'learning_rate': 0.5,
#               'max_depth': 7,
#               'gamma': 1,
#               'min_chind_weight': 1,
#               'subsample': 1,
#               'colsample_bytree': 1,
#               'colsample_bylevel': 1,
#               'colsample_bynode': 1,
#               'reg_alpha': 0,
#               'reg_lambda': 1,
#               'random_state': 842,
#               #'verbose' :0,              
#               }

#2. 모델

path = './_save/joblib/'
import joblib
model = joblib.load(path + 'ml19_joblib1_save.joblib')
#model = DecisionTreeRegressor()
#model = RandomForestRegressor()
# model = XGBRegressor(    
#     #**parameters,  #  **args: 딕셔너리로 받는다. *args: 튜플로 받는다.  
#     )

# model.set_params(**parameters,
#                  early_stopping_rounds=20,
#                  )




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
# model.fit(x_train, y_train,
#           eval_set = [(x_test, y_test)],
#           verbose = 1, 
#           )

#4. 평가, 예측

result = model.score(x_test, y_test)
print('최종 점수:', result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred )
print('r2:', r2)



# path = './_save/joblib/'
# import joblib
# joblib.dump(model, path + 'ml19_joblib1_save.joblib')
##############################################    bootstrap=True      ###############################
# 최종 점수: 0.7730051619658873
# r2: 0.7730051619658873
##############################################    bootstrap=False      ###############################
# 최종 점수: 0.778563074264455
# r2: 0.778563074264455
##############################################    RandomForestRegressor   ##############################
# 최종 점수: 0.7992020065486529
# r2: 0.7992020065486529
########################################################################################################
# 최종 점수: 0.834303313332774
# r2: 0.834303313332774
##############################################   joblib으로 불러와서 평가하기     #################################
# 최종 점수: 0.8047307364581832
# r2: 0.8047307364581832