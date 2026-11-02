
# ============================================================
# CRAZY AI - Decision Tree Regression V1.0
# Dataset : California Housing
# Model   : DecisionTreeRegressor
# Validation : 5-Fold Cross Validation
# Metrics : R2, MAE, MSE, RMSE
# ============================================================

# 1. Import Libraries
import numpy as np
import pandas as pd
import time

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# ============================================================
# 2. Load Dataset
# ============================================================

start_time = time.time()

dataset = fetch_california_housing(as_frame=True)

X = dataset.data
y = dataset.target

print("=" * 60)
print("CALIFORNIA HOUSING DATASET")
print("=" * 60)

print("X Shape:", X.shape)
print("y Shape:", y.shape)
print("\nFeature Names:")
print(X.columns.tolist())

print("\nDataset Sample:")
print(X.head())

# ============================================================
# 3. Train / Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=333
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# ============================================================
# 4. Define Model
# ============================================================

model = DecisionTreeRegressor(
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=333
)

# ============================================================
# 5. K-Fold Cross Validation
# ============================================================

n_split = 5

kfold = KFold(
    n_splits=n_split,
    shuffle=True,
    random_state=333
)

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=kfold,
    scoring='r2',
    n_jobs=-1
)

print("\n" + "=" * 60)
print("5-FOLD CROSS VALIDATION")
print("=" * 60)

for i, score in enumerate(cv_scores):
    print(f"Fold {i+1} R2 Score: {score:.4f}")

print(f"\nMean CV R2 : {np.mean(cv_scores):.4f}")
print(f"Std CV R2  : {np.std(cv_scores):.4f}")

# ============================================================
# 6. Train Final Model
# ============================================================

model.fit(X_train, y_train)

# ============================================================
# 7. Prediction
# ============================================================

y_pred = model.predict(X_test)

# ============================================================
# 8. Model Evaluation
# ============================================================

r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

print(f"R2 Score : {r2:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"MSE      : {mse:.4f}")
print(f"RMSE     : {rmse:.4f}")

# ============================================================
# 9. Feature Importance
# ============================================================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(importance.to_string(index=False))

# ============================================================
# 10. Actual vs Predicted
# ============================================================

result = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

result['Error'] = result['Actual'] - result['Predicted']

print("\nPrediction Results:")
print(result.head(10))

# ============================================================
# 11. Execution Time
# ============================================================

end_time = time.time()

print("\nExecution Time:", round(end_time - start_time, 2), "seconds")

print("\nCRAZY AI Decision Tree Regression Completed!")

################################################################################################################################

# ============================================================
# CALIFORNIA HOUSING DATASET
# ============================================================
# X Shape: (20640, 8)
# y Shape: (20640,)

# Feature Names:
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

# Dataset Sample:
#    MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude
# 0  8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88    -122.23
# 1  8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86    -122.22
# 2  7.2574      52.0  8.288136   1.073446       496.0  2.802260     37.85    -122.24
# 3  5.6431      52.0  5.817352   1.073059       558.0  2.547945     37.85    -122.25
# 4  3.8462      52.0  6.281853   1.081081       565.0  2.181467     37.85    -122.25

# Train Shape: (16512, 8)
# Test Shape : (4128, 8)

# ============================================================
# 5-FOLD CROSS VALIDATION
# ============================================================
# Fold 1 R2 Score: 0.7079
# Fold 2 R2 Score: 0.7037
# Fold 3 R2 Score: 0.7294
# Fold 4 R2 Score: 0.7048
# Fold 5 R2 Score: 0.6974

# Mean CV R2 : 0.7086
# Std CV R2  : 0.0109

# ============================================================
# FINAL MODEL EVALUATION
# ============================================================
# R2 Score : 0.6998
# MAE      : 0.4073
# MSE      : 0.3825
# RMSE     : 0.6184

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
#    Feature  Importance
#     MedInc    0.578441
#   AveOccup    0.142118
#   Latitude    0.090456
#  Longitude    0.084098
#   HouseAge    0.041645
#   AveRooms    0.038428
# Population    0.014271
#  AveBedrms    0.010542

# Prediction Results:
#     Actual  Predicted     Error
# 0  1.87500   1.521400  0.353600
# 1  1.56300   1.962355 -0.399355
# 2  2.54500   2.006769  0.538231
# 3  0.66500   0.622673  0.042327
# 4  3.43000   2.849077  0.580923
# 5  1.56700   1.591840 -0.024840
# 6  3.84800   2.472000  1.376000
# 7  1.23400   1.202167  0.031833
# 8  1.03200   1.011750  0.020250
# 9  5.00001   3.317200  1.682810

# Execution Time: 2.48 seconds

# CRAZY AI Decision Tree Regression Completed!