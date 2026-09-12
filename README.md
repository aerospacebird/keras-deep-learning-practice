# Keras Deep Learning Practice

Keras / TensorFlow 딥러닝 실습 코드 모음입니다.  
회귀(Regression)와 분류(Classification) 문제를 중심으로 기초부터 모델 저장/로드까지 단계적으로 학습한 코드들을 정리했습니다.

## 📂 Repository Structure

### 1. 기초 (Basic Dense / MLP)
| 파일 | 설명 |
|------|------|
| `keras01.py` ~ `keras04.py` | Sequential + Dense 기본 모델 |
| `keras06_batch.py` | batch_size 사용 |
| `keras07_행렬.py` | NumPy 행렬 shape / 행렬 곱셈 개념 |
| `keras08_mlp*.py` | Multi-Layer Perceptron (다차원 입력) |

### 2. Train / Test / Validation Split
| 파일 | 설명 |
|------|------|
| `keras09_train_test*.py` | train_test_split 수동/자동 분할 |
| `keras16_validation*.py` | validation_data / validation_split |
| `keras17_val*.py` | California, Diabetes, Boston, 따릉이, Kaggle Bike validation 적용 |

### 3. 과적합 방지 & EarlyStopping
| 파일 | 설명 |
|------|------|
| `keras19_overfit*.py` | loss / val_loss 시각화로 과적합 확인 |
| `keras20_EarlyStopping*.py` | EarlyStopping 콜백 적용 |

### 4. 스케일링 (Scaling)
| 파일 | 설명 |
|------|------|
| `keras27_scaler01_california.py` | MinMaxScaler |
| `keras28_scaler*.py` | MinMax / Standard / MaxAbs / RobustScaler 비교 |

### 5. 회귀 (Regression)
- California Housing
- Diabetes
- Boston Housing
- Dacon 따릉이 (`keras13_ddarung*.py`)
- Kaggle Bike Sharing Demand (`keras14_kaggle_bike1.py` 등)

### 6. 분류 (Classification)
| 파일 | 설명 |
|------|------|
| `keras21_sigmoid_metrics_cancer.py` | 이진 분류 (Breast Cancer) + sigmoid |
| `keras22_sigmoid_santander.py` | Kaggle Santander 이진 분류 |
| `keras23_softmax*.py` | 다중 분류 (Iris, Wine, Covtype, Digits) + One-Hot Encoding |
| `keras24_kaggle_santander.py` | Santander softmax 버전 |

### 7. 모델 저장 / 로드
| 파일 | 설명 |
|------|------|
| `keras29_1_save_model.py` | 모델 구조 저장 (`.keras`) |
| `keras29_2_load_model.py` | 저장된 모델 로드 |
| `keras29_3_save_model2.py` | 학습 후 모델 저장 |
| `keras29_4_load_model2.py` | 학습된 가중치 포함 모델 로드 |

### 8. 기타
- `keras15_verbose.py` : verbose 옵션
- `keras18__time.py` : 학습 시간 측정
- `keras25_summary.py` : model.summary() + Parameter 계산
- `keras26_input_shape.py` : input_shape 사용법

## 🛠 주요 기술 스택
- **Framework**: TensorFlow / Keras
- **Data**: scikit-learn datasets, Pandas, NumPy
- **Preprocessing**: MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
- **Callbacks**: EarlyStopping
- **Metrics**: MSE, RMSE, R², Accuracy, Binary/Categorical Crossentropy

## 📌 사용 방법
```bash
# 예시
python keras20_EarlyStopping1_california.py
```

대부분의 파일은 독립적으로 실행 가능합니다.  
일부 Kaggle / Dacon 데이터는 로컬 경로(`./_data/...`)를 사용하므로 데이터 파일을 해당 위치에 준비해야 합니다.

## 📅 업로드 일자
2026-09-12

---
**Author**: [aerospacebird](https://github.com/aerospacebird)
