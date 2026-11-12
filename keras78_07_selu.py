
#2017 구글 에서 swish 발표.
# x에 sigmoid() 곱한 함수 
# Sigmoid Liner Unit

import numpy as np
import matplotlib.pyplot as plt


x = np.arange(-5, 5, 0.1)

#SELU 고정상수 (논문 제안값)

alpha = 1.6732632423543772
lmbda = 1.0507009873554805


# def selu(x, alpha, lmbda):
#     return lmbda*((x>0)*x + (x<=0)*(alpha*(np.exp(x) -1)))



selu = lambda x, alpha, lambda : lmbda*((x>0)*x + (x<=0)*(alpha*(np.exp(x) -1)))
y = selu(x, alpha, lmbda)

# def mish(x):
#     return x * np.tanh(np.log(1+ np.exp(x)))

# def elu(x, alpha):
#     return (x>0)*x + (x<0)*(alpha*(np.exp(x)-1))  # alpha  parameter

# exit()
# silu = lambda x : x * (1 / (1+np.exp(-x)))
# def silu(x):
#     return x * (1 / (1+np.exp(-x)))


#relu = lambda x: np.maximum(0, x)
#y = silu(x)

#silu = lambda x: x * (1 / (1+np.exp(-x)))


#mish = lambda x : x * np.tanh(np.log(1+ np.exp(x)))
# elu = lambda x, alpha : (x>0)*x + (x<0)*(alpha*(np.exp(x)-1))

# y = elu(x, 1)

plt.plot(x, y)
plt.grid()
plt.show()
