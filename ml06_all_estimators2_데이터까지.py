

# ============================================================
# CRAZY AI Regression Model Benchmark V1.0
# All Scikit-learn Regressors + K-Fold Cross Validation
# ============================================================

import numpy as np
import pandas as pd
import sklearn as sk
import time
import warnings

from sklearn.datasets import (
    load_diabetes,
    fetch_california_housing
)

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from sklearn.utils import all_estimators

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. Environment
# ------------------------------------------------------------

print("=" * 65)
print("CRAZY AI REGRESSION MODEL BENCHMARK V1.0")
print("=" * 65)

print("Scikit-learn Version:", sk.__version__)

# ------------------------------------------------------------
# 2. Dataset Loading
# ------------------------------------------------------------

diabetes = load_diabetes(return_X_y=True)

california = fetch_california_housing(return_X_y=True)

datasets = [
    diabetes,
    california
]

data_name = [
    "Diabetes",
    "California Housing"
]

# ------------------------------------------------------------
# 3. Load All Regressors
# ------------------------------------------------------------

all_models = all_estimators(type_filter="regressor")

print("\nTotal Regressor Classes:", len(all_models))

# ------------------------------------------------------------
# 4. K-Fold Configuration
# ------------------------------------------------------------

n_split = 5

kfold = KFold(
    n_splits=n_split,
    shuffle=True,
    random_state=333
)

# ------------------------------------------------------------
# 5. Benchmark Execution
# ------------------------------------------------------------

start = time.time()

all_results = {}

for index, dataset in enumerate(datasets):

    X, y = dataset

    print("\n" + "=" * 65)
    print("DATASET:", data_name[index])
    print("=" * 65)

    print("X Shape:", X.shape)
    print("Y Shape:", y.shape)

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=333
    )

    model_results = []
    failed_models = []

    # --------------------------------------------------------
    # 6. Model Loop
    # --------------------------------------------------------

    for name, estimator_class in all_models:

        try:

            # Instantiate model
            estimator = estimator_class()

            # Scaling + Model Pipeline
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", estimator)
            ])

            # ------------------------------------------------
            # 7. K-Fold Cross Validation
            # ------------------------------------------------

            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=kfold,
                scoring="r2",
                n_jobs=1,
                error_score="raise"
            )

            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)

            # ------------------------------------------------
            # 8. Final Model Training
            # ------------------------------------------------

            model.fit(X_train, y_train)

            # Test Prediction
            y_pred = model.predict(X_test)

            # ------------------------------------------------
            # 9. Evaluation
            # ------------------------------------------------

            test_r2 = r2_score(y_test, y_pred)

            rmse = np.sqrt(
                mean_squared_error(y_test, y_pred)
            )

            mae = mean_absolute_error(
                y_test,
                y_pred
            )

            model_results.append({
                "Model": name,
                "CV_R2_Mean": cv_mean,
                "CV_R2_STD": cv_std,
                "Test_R2": test_r2,
                "RMSE": rmse,
                "MAE": mae
            })

        except Exception as e:

            failed_models.append({
                "Model": name,
                "Error": str(e)[:200]
            })

    # --------------------------------------------------------
    # 10. Results
    # --------------------------------------------------------

    results_df = pd.DataFrame(model_results)

    if results_df.empty:
        print("No models completed successfully.")
        continue

    # Rank by Cross Validation R2
    results_df = results_df.sort_values(
        by="CV_R2_Mean",
        ascending=False
    ).reset_index(drop=True)

    all_results[data_name[index]] = results_df

    print("\nSuccessful Models:", len(results_df))
    print("Failed Models:", len(failed_models))

    print("\nTOP 10 REGRESSION MODELS")
    print("-" * 100)

    print(
        results_df.head(10).to_string(index=False)
    )

    # --------------------------------------------------------
    # 11. Best Model
    # --------------------------------------------------------

    best_model = results_df.iloc[0]

    print("\n" + "=" * 65)
    print("BEST MODEL:", data_name[index])
    print("=" * 65)

    print("Model       :", best_model["Model"])
    print("CV Mean R2  :", round(best_model["CV_R2_Mean"], 4))
    print("CV Std      :", round(best_model["CV_R2_STD"], 4))
    print("Test R2     :", round(best_model["Test_R2"], 4))
    print("Test RMSE   :", round(best_model["RMSE"], 4))
    print("Test MAE    :", round(best_model["MAE"], 4))

    # --------------------------------------------------------
    # 12. Failed Models
    # --------------------------------------------------------

    if failed_models:

        failed_df = pd.DataFrame(failed_models)

        print("\nFAILED MODEL LIST")
        print(failed_df.to_string(index=False))

# ------------------------------------------------------------
# 13. Execution Time
# ------------------------------------------------------------

end = time.time()

print("\n" + "=" * 65)
print("BENCHMARK COMPLETED")
print("=" * 65)

print("Execution Time:", round(end - start, 2), "seconds")
print("Scikit-learn:", sk.__version__)




# import numpy as np
# from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, load_diabetes, fetch_california_housing
# from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score

# from sklearn.utils import all_estimators
# import sklearn as sk
# import time
# import warnings
# warnings.filterwarnings('ignore')


# #1. 데이터
# iris = load_iris(return_X_y=True)
# cancer = load_breast_cancer(return_X_y=True)
# wine = load_wine(return_X_y=True)
# digits = load_digits(return_X_y=True)

# # datasets = [iris, cancer, wine, digits]
# # data_name = ['아이리스', '캔서', '와인', '디지트']

# datasets = [diabetes, california]
# data_name = ['당뇨', '캘리포니아']

# #2. 모델
# #all = all_estimators(type_filter='classifier')
# all = all_estimators(type_filter='regressor')

# n_split = 5
# kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)

# start = time.time()
# for index, value in enumerate(datasets):
#     x, y = value
    
#     x_train, x_test, y_train, y_test = train_test_split(
#         x, y, shuffle=True, random_state=333, train_size=0.8,
#         stratify=y,
#     )
#     scaler = StandardScaler()
#     x_train = scaler.fit_transform(x_train)
#     x_test = scaler.transform(x_test)

#     print("===========", data_name[index], "===========")
#     model_name=[]
#     model_acc=[]

#     for name, model in all:
#         # print(name)
#         try:
#             model = model()

#             #3. 훈련
#             # model.fit(x_train, y_train)
#             scores = cross_val_score(model, x_train, y_train, cv=kfold)

#             #4. 평가
#             y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)

#             acc = np.round(accuracy_score(y_test, y_pred),4)
#             model_name.append(name)
#             model_acc.append(acc)

#         except:  # noqa: E722, S112
#             # print(name, '모델 오류')
#             continue

#     model_name = np.array(model_name)
#     model_acc = np.array(model_acc)
#     max_index = np.argmax(model_acc)
#     print("최고 모델 : ", model_name[max_index], "ACC :", model_acc[max_index])
# end = time.time()

# print("걸린시간 : ", round(end - start, 2), "초")