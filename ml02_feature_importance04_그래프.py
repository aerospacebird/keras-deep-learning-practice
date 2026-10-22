from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


#1. data
x, y = load_iris(return_X_y=True)
print(x.shape, y.shape) # (150,4) (150,)

random_state = 12313

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=random_state,
    stratify=y #회기에서 쓸 수 없음
)


#2.model

model = XGBClassifier(random_state=random_state)


#3.compile,train
model.fit(x_train,y_train)
print("====================", model.__class__.__name__, "===============")


#4.평가 ,예측
score = model.score(x_test,y_test)
print("acc : " ,score)
print(model.feature_importances_)

#F스코어 : 빈도수
gain_score = model.get_booster().get_score(importance_type='gain') #손실함수의 개선율
print("gain_score", gain_score)
#Gain 스코어 : 손실함수의 개선 점수. 높으면 개선 많이 됬다.

import matplotlib.pyplot as plt
from xgboost import plot_importance

# #F-score 그래프
# plot_importance(model, importance_type='weight')
# plt.title("F-Score")
# plt.show()

#Gain-score 그래프
plot_importance(model, importance_type='gain')
plt.title("Gain-score")
plt.show()


