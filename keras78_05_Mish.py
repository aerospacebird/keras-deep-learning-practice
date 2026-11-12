#2017 구글 에서 swish 발표.
# x에 sigmoid() 곱한 함수 
# Sigmoid Liner Unit

import numpy as np
import matplotlib.pyplot as plt


x = np.arange(-5, 5, 0.1)

def mish(x):
    return x * np.tanh(np.log(1+ np.exp(x)))



# exit()
# silu = lambda x : x * (1 / (1+np.exp(-x)))
# def silu(x):
#     return x * (1 / (1+np.exp(-x)))


#relu = lambda x: np.maximum(0, x)
#y = silu(x)

#silu = lambda x: x * (1 / (1+np.exp(-x)))


mish = lambda x : x * np.tanh(np.log(1+ np.exp(x)))

y = mish(x)
plt.plot(x, y)
plt.grid()
plt.show()
