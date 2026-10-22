

from sklearn.datasets import load_iris, load_diabetes
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor


#1. 데이터

x, y = load_iris(return_X_y=True)
print(x.shape, y.shape) # (150, 4) (150,)

random_state=12313

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=random_state,
    #stratify=y,
    )

#2. 모델 구성
model1 = DecisionTreeRegressor(random_state=random_state)
model2 = RandomForestRegressor(random_state=random_state)
model3 = GradientBoostingRegressor(random_state=random_state)
model4 = XGBRegressor(random_state=random_state)


models = [model1, model2, model3, model4]

#3. 컴파일, 훈련
for model in models:
    model.fit(x_train, y_train)
    print("=========================", model.__class__.__name__, " =======================")

    #model1.fit(x_train, y_train)

    #4. 평가, 예측
    score = model.score(x_test, y_test)
    print(" r2 :", score) # 0.9666666666666667, #0.9333333333333333, # 0.9666666666666667, # 1.0 : 
    print(model.feature_importances_)

############################################ Regressor ########################################################

# ========================= DecisionTreeRegressor  =======================
#  r2 : 1.0
# [0.         0.00625    0.79079612 0.20295388]
# ========================= RandomForestRegressor  =======================
#  r2 : 0.99777
# [0.00913611 0.0070173  0.57807351 0.40577308]
# ========================= GradientBoostingRegressor  =======================
#  r2 : 0.9946670163008692
# [0.00765128 0.0035671  0.62755951 0.36122211]
# ========================= XGBRegressor  =======================
#  r2 : 0.9974752068519592
# [0.00295137 0.00990401 0.94706464 0.04007994]

#########################################################################################################
"""
F1 Score란?이진 분류에서 정밀도(Precision) 와 재현율(Recall) 의 조화평균입니다.$$\text{Precision} = \frac{TP}{TP+FP}, \quad
\text{Recall} = \frac{TP}{TP+FN}, \quad
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
























용어의미 (강수 예)TP비 온다고 예측 · 실제로 비FP비 온다고 예측 · 실제로는 맑음FN맑다고 예측 · 실제로 비TN맑다고 예측 · 실제로 맑음
0 ~ 1 (1에 가까울수록 좋음)
클래스 불균형(맑은 날이 훨씬 많음)일 때 정확도보다 유용
다중 클래스면 macro-F1(클래스별 F1 평균) 또는 weighted-F1을 씁니다.여행 기상 모델에 적용하는 방법기온(℃)은 회귀 → MAE/RMSE.
F1을 쓰려면 예를 들어:
강수 유무: rn_day > 0 → 비(1) / 맑음(0)
기온 구간: 평년 대비 낮음 / 평년 / 높음
여행 적합: 강수확률·기온으로 Comfort / Caution / Poor



"""

