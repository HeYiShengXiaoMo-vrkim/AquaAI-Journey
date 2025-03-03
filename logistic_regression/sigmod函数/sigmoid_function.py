import numpy as np
import math
import matplotlib.pyplot as plt

def sigmoid(x):
    a = []
    for item in x:
        a.append(1.0/(1.0+math.exp(-item)))
    return a

x = np.arange(-10,10,0.1)  # 生成-10到10的数组，步长为0.1
y = sigmoid(x)
plt.plot(x, y)  # 绘制x和y的图像
plt.show()  # 显示图像