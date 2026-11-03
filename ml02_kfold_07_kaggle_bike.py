
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error



#1. 데이터
path = './_data/kaggle_bike/'  # 상대경로

#path = "c:/study/_data/ddarung/"  # 절대경로
#path = "c:/study/_data/ddarung/"  # 슬래시 역슬래시 상관없어
#path = "c://study//_data//ddarung/"  # 슬래시 2개도 가능하다.  가독성있게 작성하라.

train_csv = pd.read_csv(path + "train.csv", index_col=0) #1459 rows x 11columns ------->>>>#1459 rows x 10 columns index_col=0 때문이다.
print(train_csv) #10886rows x 11 columns

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv) #[6493 rows x 8 columns]

submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)
print(submission) #[6493 rows x 1 columns]

print(train_csv.shape) #(10886, 11)
print(test_csv.shape) # (6493, 8)
print(submission.shape) # (6493, 1)



print(train_csv.columns) 

print(train_csv.info())
print(test_csv.info())

print(train_csv.describe()) 

################################### 결측치 확인 ###########################################################################
print(train_csv.isna().sum())
print(test_csv.isnull().sum())

###################################### x, y 분리 ###########################################################################

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)   #[10886 rows x 8 columns]

y = train_csv['count']
print(y)    
print(y.shape)    #(10886, )


# from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                 train_size=0.75,
                 #test_size=0.3,
                 shuffle=True,   # 디폴트 섞는다. 75%/25% 비율로 섞는다.
                 random_state=500,   # 42  디폴트 값이다. 이값을 변경해도 성능에 영향을 줄수가 있다.                
)


# ==================================================
# Train / Validation 분리
# ==================================================
# 현재 train 데이터의 25%를 validation으로 사용

x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    train_size=0.75,
    shuffle=True,
    random_state=500,
)
# Validation을 만들려면 **두 번째 train_test_split()**을 사용해야 합니다.
####################################### submit 물밑작업 #######################################################

print(test_csv.info())

###################################결측치 처리 2.  평균값 넣기     #################################################
#test_csv = test_csv.fillna(test_csv.mean())
#print(test_csv.info())  # (6493, 8)
#print(test_csv.shape)  # (6493, 8)
#1. data

# x, y = load_iris(return_X_y=True)
# print(x, y)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

#2. model

model = DecisionTreeClassifier()


#3. Train
scores = cross_val_score(model, x, y, cv=kfold)

print('acc :', scores, '\n평균 acc : ', round(np.mean(scores),2))

####################################################################################################################################################

# acc : [0.00734619 0.00826826 0.00505282 0.00551217 0.01148369] 
# 평균 acc :  0.01

######################################################################################################################################################


