
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

aaa = np.array([[-10, 2,3,4,5,6,7,8,9,10,11,12,50],
               [100, 200,-30,400,500,600, -70000,800,900,1000,210,420,350] 
                
                
                ]).T

processed_aaa = aaa.astype(float)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for i, data in enumerate(aaa.T, start=1):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0]
    outlier_values = data[outlier_indices]
    processed_data = np.clip(data, lower_bound, upper_bound)
    processed_aaa[:, i - 1] = processed_data

    print(f'{i}번째 리스트의 이상치 위치: {outlier_indices}')
    print(f'{i}번째 리스트의 이상치 값: {outlier_values}')
    print(f'{i}번째 리스트 처리 후: {processed_data}')

    axes[0, i - 1].boxplot(data)
    axes[0, i - 1].set_title(f'{i}번째 리스트 - 처리 전')
    axes[0, i - 1].set_ylabel('값')
    axes[0, i - 1].grid(True)

    axes[1, i - 1].boxplot(processed_data)
    axes[1, i - 1].set_title(f'{i}번째 리스트 - 처리 후')
    axes[1, i - 1].set_ylabel('값')
    axes[1, i - 1].grid(True)

print('이상치 처리 결과 배열:')
print(processed_aaa)
plt.tight_layout()
plt.show()
