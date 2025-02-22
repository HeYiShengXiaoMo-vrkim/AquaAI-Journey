import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
# 根据正态分布初始化数据
x1 = 2*np.random.rand(100, 1)
x2 = 2*np.random.rand(100, 1)
x = np.c_[x1, x2]

y = 4 + 3*x1 + 5*x2 + np.random.randn(100, 1)

reg = LinearRegression()  # 拟合模型,创建一个回归的对象
reg.fit(x, y)  # 拟合模型
print(reg.intercept_, reg.coef_)  # 输出系数

x_new = np.array([[0, 0], [2, 2],  [2, 4]])
y_predict = reg.predict(x_new)

# 绘图进行展示真实的数据点和我们预测用的模型
plt.plot(x_new[:, 0], y_predict, "r-")
plt.plot(x1, y, "b.")
plt.axis((0.0, 2.0, 0.0, 15.0))
plt.show()

