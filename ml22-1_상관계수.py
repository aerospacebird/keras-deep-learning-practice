
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
#1. 데이터

datasets = load_iris()
print(datasets.feature_names)

x = datasets['data']
y = datasets.target


df = pd.DataFrame(x, columns=datasets.feature_names)
print(df)

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

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=444,
    stratify=y,
)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


parameters = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 7,
    'subsample': 1,
    'colsample_bytree': 1,
    'random_state': 842,
}

#2. 모델
model = XGBClassifier(
    **parameters,
    early_stopping_rounds=20,
)

#3. 훈련
model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)],
          verbose = 1, 
          )

#4. 평가, 예측

result = model.score(x_test, y_test)
print('최종 점수:', result)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print('accuracy:', accuracy)



# path = './_save/joblib/'
# import joblib

# joblib.dump(model, path + 'ml22_joblib2_save.joblib')

path = './_save/pickle/'
os.makedirs(path, exist_ok=True)
with open(path + 'ml22_pickle2_save.pickle', 'wb') as file:
    pickle.dump(model, file)