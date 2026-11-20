from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

# 1. 데이터: XOR은 두 입력 중 하나만 1일 때 결과가 1입니다.
# 행별 입력: [0, 0], [0, 1], [1, 0], [1, 1]
# 정답 레이블:   0,       1,       1,       0
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([0, 1, 1, 0])

# 2. 모델: RBF 커널은 XOR처럼 직선 하나로 나눌 수 없는 데이터를 분류합니다.
model = SVC(kernel='rbf')

# 3. 훈련: 입력 데이터와 정답 레이블로 분류 경계를 학습합니다.
model.fit(x_data, y_data)

# 4. 예측: 학습한 모델로 입력 데이터의 레이블을 예측합니다.
y_predict = model.predict(x_data)

# 정답 레이블과 예측값을 비교해 정확도를 계산합니다.
# 여기서는 학습에 사용한 데이터로 평가하므로, 새 데이터 성능을 뜻하지는 않습니다.
acc = accuracy_score(y_data, y_predict)
print('acc:', acc)
print("accuracy_score :", acc)