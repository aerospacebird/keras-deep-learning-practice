import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer #IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
#from sklearn.experimental import enable_iterative_imputer



data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10, 12],
                     [np.nan, 4, np.nan, 8, np.nan, 8],
                     ]).T
print(data)
data.columns =['x1', 'x2', 'x3', 'x4']
print(data)

from xgboost import XGBRegressor

xgb = XGBRegressor(
    max_depth=5,
    learning_rate=0.1,
    random_state=333,
)

imputer = IterativeImputer(
    estimator=xgb, 
    max_iter=10,
    random_state=333,
)

data2 = imputer.fit_transform(data)
print(data2)

print("실제 반복횟수 :", imputer.n_iter_)

##############################################################################################################
# [[ 2.          2.          2.          4.01578808]
#  [ 2.02664208  4.          4.          4.        ]
#  [ 6.          4.0039463   6.          4.01578808]
#  [ 8.          8.          8.          8.        ]
#  [10.          7.98026466 10.          7.99865484]
#  [ 9.97863579  7.98026466 12.          8.        ]]
# 실제 반복횟수 : 2

