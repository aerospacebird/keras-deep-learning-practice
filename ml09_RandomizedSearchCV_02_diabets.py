
import numpy as np
import time
import warnings

from sklearn.datasets import load_diabetes
from sklearn.model_selection import (
    train_test_split,
    KFold,
    RandomizedSearchCV
)
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from xgboost import XGBRegressor

warnings.filterwarnings('ignore')


# ============================================================
# 1. DATA PREPARATION
# ============================================================

x, y = load_diabetes(return_X_y=True)

print("X shape:", x.shape)
print("y shape:", y.shape)
print("Target type: Regression")

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle=True,
    random_state=333,
    train_size=0.8
)

# Regression: KFold (NOT StratifiedKFold)
n_splits = 5

kfold = KFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=333
)


# ============================================================
# 2. HYPERPARAMETER SEARCH SPACE
# ============================================================

parameters = [
    {
        'n_estimators': [100, 500],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.1, 0.01, 0.001]
    },
    {
        'max_depth': [3, 5, 8, 10],
        'learning_rate': [0.1, 0.01, 0.001]
    },
    {
        'min_child_weight': [2, 3, 5, 10],
        'learning_rate': [0.1, 0.01, 0.001]
    }
]


# ============================================================
# 3. MODEL
# ============================================================

xgb = XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    random_state=333,
    n_jobs=1
)

model = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=parameters,
    n_iter=10,
    cv=kfold,
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=-1,
    random_state=333,
    refit=True
)


# ============================================================
# 4. TRAINING
# ============================================================

start = time.perf_counter()

model.fit(x_train, y_train)

end = time.perf_counter()


# ============================================================
# 5. BEST MODEL AND PARAMETERS
# ============================================================

print("\n========== BEST MODEL ==========")

print("Best Estimator:")
print(model.best_estimator_)

print("\nBest Parameters:")
print(model.best_params_)

# Convert negative RMSE into positive RMSE
print(
    "\nBest CV RMSE:",
    round(-model.best_score_, 4)
)


# ============================================================
# 6. EVALUATION AND PREDICTION
# ============================================================

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

mae = mean_absolute_error(y_test, y_pred)


# ============================================================
# 7. FINAL RESULTS
# ============================================================

print("\n========== TEST RESULTS ==========")

print("R2 Score :", round(r2, 4))
print("RMSE     :", round(rmse, 4))
print("MAE      :", round(mae, 4))

print(
    "\nTraining Time:",
    round(end - start, 2),
    "seconds"
)

print("\n========== PREDICTION COMPARISON ==========")

for i in range(min(10, len(y_test))):
    print(
        f"Actual: {y_test[i]:.2f} | "
        f"Predicted: {y_pred[i]:.2f}"
    )

############################################## ml09_RandomizedSearchCV_02_diabets ########################################
# Fitting 5 folds for each of 10 candidates, totalling 50 fits

# ========== BEST MODEL ==========
# Best Estimator:
# XGBRegressor(base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=None, colsample_bynode=None,
#              colsample_bytree=None, device=None, early_stopping_rounds=None,
#              enable_categorical=False, eval_metric='rmse', feature_types=None,
#              feature_weights=None, gamma=None, grow_policy=None,
#              importance_type=None, interaction_constraints=None,
#              learning_rate=0.1, max_bin=None, max_cat_threshold=None,
#              max_cat_to_onehot=None, max_delta_step=None, max_depth=3,
#              max_leaves=None, min_child_weight=None, missing=nan,
#              monotone_constraints=None, multi_strategy=None, n_estimators=None,
#              n_jobs=1, num_parallel_tree=None, ...)

# Best Parameters:
# {'max_depth': 3, 'learning_rate': 0.1}

# Best CV RMSE: 58.843

# ========== TEST RESULTS ==========
# R2 Score : 0.3142
# RMSE     : 60.2975
# MAE      : 49.9491

# Training Time: 5.99 seconds

# ========== PREDICTION COMPARISON ==========
# Actual: 202.00 | Predicted: 108.81
# Actual: 168.00 | Predicted: 122.73
# Actual: 94.00 | Predicted: 93.74
# Actual: 85.00 | Predicted: 151.61
# Actual: 259.00 | Predicted: 142.00
# Actual: 39.00 | Predicted: 77.03
# Actual: 245.00 | Predicted: 146.87
# Actual: 171.00 | Predicted: 191.27
# Actual: 95.00 | Predicted: 152.49
# Actual: 137.00 | Predicted: 195.76
# PS C:\study> 
################################################################################################################################
# import numpy as np
# from sklearn.datasets import load_digits, load_diabetes
# from sklearn.model_selection import train_test_split,StratifiedKFold
# from sklearn.ensemble import HistGradientBoostingClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from xgboost import XGBClassifier
# import time
# import warnings
# warnings.filterwarnings('ignore')



# #1. 데이터 준비

# x, y = load_diabetes(return_X_y=True)

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, shuffle=True, random_state=333, train_size=0.8,
#     stratify=y

# )

# n_splits = 5
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

# parameters = [
#      {'n_estimators': [100, 500], 'max_depth' : [6, 10, 12],
#      'learning_rate': [0.1, 0.01, 0.001]}, #18
#      {'max_depth' : [6, 8, 10, 12], 'learning_rate': [0.1, 0.01, 0.001]}, #12
#      {'min_child_weight':[2,3,5,10], 'learning_rate' : [0.1, 0.01, 0.001]} #12

# ]


# #2. 모델

# xgb = XGBClassifier()
# model = RandomizedSearchCV(xgb, parameters, cv=kfold,
#              verbose=1,
#              n_jobs=-1,            
#              )

# #3. 훈련
# start = time.perf_counter()
# model.fit(x_train, y_train)
# end= time.perf_counter()

# print("최적의 매개변수 :", model.best_estimator_)
# print("최적의 파라미터 :", model.best_params_)

# #4. 평가, 예측
# print('model.score :', model.score(x_test, y_test))
# print('best_score :', model.best_score_)

# print("걸린시간 :", round(end-start, 2), "초")

# ####################################################################################################################################
# # 최적의 파라미터 : {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 500}
# # model.score : 0.9472222222222222
# # best_score : 0.9610288424312815
# # 걸린시간 : 21.1 초
# ######################################################################################################################################
# # 최적의 매개변수 : XGBClassifier(base_score=None, booster=None, callbacks=None,
# #               colsample_bylevel=None, colsample_bynode=None,
# #               colsample_bytree=None, device=None, early_stopping_rounds=None,
# #               enable_categorical=True, eval_metric=None, feature_types=None,
# #               feature_weights=None, gamma=None, grow_policy=None,
# #               importance_type=None, interaction_constraints=None,
# #               learning_rate=0.1, max_bin=None, max_cat_threshold=None,
# #               max_cat_to_onehot=None, max_delta_step=None, max_depth=6,
# #               max_leaves=None, min_child_weight=None, missing=nan,
# #               monotone_constraints=None, multi_strategy=None, n_estimators=500,
# #               n_jobs=None, num_parallel_tree=None, ...)
# # 최적의 파라미터 : {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 500}
# # model.score : 0.9472222222222222
# # best_score : 0.9610288424312815
# # 걸린시간 : 19.87 초
# # PS C:\study> 
# ##################################################m    RandomizedSearch_CV00             ###################################################
# # Fitting 5 folds for each of 10 candidates, totalling 50 fits
# # 최적의 매개변수 : XGBClassifier(base_score=None, booster=None, callbacks=None,
# #               colsample_bylevel=None, colsample_bynode=None,
# #               colsample_bytree=None, device=None, early_stopping_rounds=None,
# #               enable_categorical=False, eval_metric=None, feature_types=None,
# #               feature_weights=None, gamma=None, grow_policy=None,
# #               importance_type=None, interaction_constraints=None,
# #               learning_rate=0.1, max_bin=None, max_cat_threshold=None,
# #               max_cat_to_onehot=None, max_delta_step=None, max_depth=12,
# #               max_leaves=None, min_child_weight=None, missing=nan,
# #               monotone_constraints=None, multi_strategy=None, n_estimators=100,
# #               n_jobs=None, num_parallel_tree=None, ...)
# # 최적의 파라미터 : {'n_estimators': 100, 'max_depth': 12, 'learning_rate': 0.1}
# # model.score : 0.9416666666666667
# # best_score : 0.958947928765002
# #############################################################################################################################################

# # Fitting 5 folds for each of 10 candidates, totalling 50 fits
# # 최적의 매개변수 : XGBClassifier(base_score=None, booster=None, callbacks=None,
# #               colsample_bylevel=None, colsample_bynode=None,
# #               colsample_bytree=None, device=None, early_stopping_rounds=None,
# #               enable_categorical=False, eval_metric=None, feature_types=None,
# #               feature_weights=None, gamma=None, grow_policy=None,
# #               importance_type=None, interaction_constraints=None,
# #               learning_rate=0.1, max_bin=None, max_cat_threshold=None,
# #               max_cat_to_onehot=None, max_delta_step=None, max_depth=12,
# #               max_leaves=None, min_child_weight=None, missing=nan,
# #               monotone_constraints=None, multi_strategy=None, n_estimators=500,
# #               n_jobs=None, num_parallel_tree=None, ...)
# # 최적의 파라미터 : {'n_estimators': 500, 'max_depth': 12, 'learning_rate': 0.1}
# # model.score : 0.9444444444444444
# # best_score : 0.9575542005420056
# # 걸린시간 : 8.03 초