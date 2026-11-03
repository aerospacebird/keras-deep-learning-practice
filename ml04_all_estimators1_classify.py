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



# * 1. 데이터
x, y = load_iris(return_X_y=True)
# print(x, y)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle=True,
    random_state=333,
    train_size=0.8,
    stratify=y,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

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

#exit()









# # * 3, 4 훈련, 평가
# scores = cross_val_score(model, x_train, y_train, cv=kfold)

# print("acc :", scores, "\n평균 acc :", round(np.mean(scores), 2))

# """
# # acc : [0.93333333 0.9        1.         0.96666667 0.9       ] 
# # 평균 acc : 0.94
# """

# y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
# print(y_predict)
# print(y_test)

# acc = accuracy_score(y_test, y_predict)
# print("cross_val_predict acc :", acc)

##########################################################################################################################################
# acc : [0.95833333 0.91666667 0.95833333 1.         0.95833333] 
# 평균 acc : 0.96