
import numpy as np
import pandas as pd


data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan],
                     ]).T
print(data)
data.columns =['x1', 'x2', 'x3', 'x4']
print(data)
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

print(data.isnull())
print(data.isnull().sum())
print(data.info())

#1. 결측치 삭제
print(data.dropna()) # 결측치가 있는 행 삭제
print(data.dropna(axis=1)) # 결측치가 있는 열 삭제
print(data.dropna(axis=0)) # 결측치가 있는 행 삭제

#2-1. 특정값 - 평균
means = data.mean()
print(means)
data2 = data.fillna(means)
print(data2)

#2-2. 특정값 - 중위값
med = data.median()
print(med)
data3 = data.fillna(med)
print(data3)

#2-3. 특정값 - 0, 앞값, 뒷값
data4 = data.fillna(0) 
print(data4)

#2-3-2. 특정값 - 777
data4_2 = data.fillna(777)
print(data4)

#앞값으로 채움
print(data4)

#2-4. 특정값 - ffill(앞값, 뒷값), (시계열)
data5 = data.ffill()
print(data5)

#2-5. 특정값 - bfill(통상 첫번째 결측치에 사용)
data6 = data.bfill() #뒷값으로 채움
print(data6)   #가장 마지막은 행은 채울값이 없어서 Nan
print("================================================================================================================")
################################################### 특정값 ############################################################
data['x1'].mean()
print(means)


med = data['x4'].median()
print(med)

data['x1'] = data['x1'].fillna(means)
data['x2'] = data['x2'].bfill()
data['x4'] = data['x4'].fillna(med)
print(data)

###########################################################################################################################
#       0    1     2    3
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN
#       x1     x2     x3     x4
# 0  False  False  False   True
# 1   True  False  False  False
# 2  False   True  False   True
# 3  False  False  False  False
# 4  False   True  False   True
# x1    1
# x2    2
# x3    0
# x4    3
# dtype: int64
# <class 'pandas.DataFrame'>
# RangeIndex: 5 entries, 0 to 4
# Data columns (total 4 columns):
#  #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   x1      4 non-null      float64
#  1   x2      3 non-null      float64
#  2   x3      5 non-null      float64
#  3   x4      2 non-null      float64
# dtypes: float64(4)
# memory usage: 292.0 bytes
# None
#     x1   x2   x3   x4
# 3  8.0  8.0  8.0  8.0
#      x3
# 0   2.0
# 1   4.0
# 2   6.0
# 3   8.0
# 4  10.0
#     x1   x2   x3   x4
# 3  8.0  8.0  8.0  8.0
# x1    6.500000
# x2    4.666667
# x3    6.000000
# x4    6.000000
# dtype: float64
#      x1        x2    x3   x4
# 0   2.0  2.000000   2.0  6.0
# 1   6.5  4.000000   4.0  4.0
# 2   6.0  4.666667   6.0  6.0
# 3   8.0  8.000000   8.0  8.0
# 4  10.0  4.666667  10.0  6.0
# x1    7.0
# x2    4.0
# x3    6.0
# x4    6.0
# dtype: float64
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  6.0
# 1   7.0  4.0   4.0  4.0
# 2   6.0  4.0   6.0  6.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  4.0  10.0  6.0
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  0.0
# 1   0.0  4.0   4.0  4.0
# 2   6.0  0.0   6.0  0.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  0.0  10.0  0.0
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  0.0
# 1   0.0  4.0   4.0  4.0
# 2   6.0  0.0   6.0  0.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  0.0  10.0  0.0
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  0.0
# 1   0.0  4.0   4.0  4.0
# 2   6.0  0.0   6.0  0.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  0.0  10.0  0.0
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   2.0  4.0   4.0  4.0
# 2   6.0  4.0   6.0  4.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  8.0  10.0  8.0
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  4.0
# 1   6.0  4.0   4.0  4.0
# 2   6.0  8.0   6.0  8.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN
# ================================================================================================================
# x1    6.500000
# x2    4.666667
# x3    6.000000
# x4    6.000000
# dtype: float64
# 6.0
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  6.0
# 1   NaN  4.0   4.0  4.0
# 2   6.0  8.0   6.0  6.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  6.0
# PS C:\study> 