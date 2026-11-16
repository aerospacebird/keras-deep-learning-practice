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
                     [np.nan, 4, np.nan, 8, np.nan],
                     ]).T
print(data)
data.columns =['x1', 'x2', 'x3', 'x4']
print(data)


imputer = KNNImputer(n_neighbors=2)
data2 = imputer.fit_transform(data)
print(data2)



#K - Nearest - Neighbor 






# # 0   2.0  2.0   2.0  NaN
# # 1   NaN  4.0   4.0  4.0
# # 2   6.0  NaN   6.0  NaN
# # 3   8.0  8.0   8.0  8.0
# # 4  10.0  NaN  10.0  NaN
# imputer = SimpleImputer(strategy='most_frequent')
# data2 = imputer.fit_transform(data)
# print(data2)


# imputer2 = SimpleImputer(strategy='constant', fill_value=777)
# data3 = imputer2.fit_transform(data)
# print(data3)






# imputer3 = SimpleImputer(
#     strategy=lambda x: np.max((x)                     )
#     )

# data4 = imputer3.fit_transform(data)
# print(data4)        





#########################################################################################################################
#       0    1     2    3
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN
# 5   NaN  NaN  12.0  NaN
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN
# 5   NaN  NaN  12.0  NaN
# [[ 2.  2.  2.  4.]
#  [ 2.  4.  4.  4.]
#  [ 6.  2.  6.  4.]
#  [ 8.  8.  8.  8.]
#  [10.  2. 10.  4.]
#  [ 2.  2. 12.  4.]]
# [[  2.   2.   2. 777.]
#  [777.   4.   4.   4.]
#  [  6. 777.   6. 777.]
#  [  8.   8.   8.   8.]
#  [ 10. 777.  10. 777.]
#  [777. 777.  12. 777.]]
# PS C:\study> 
###################################################################################################################

