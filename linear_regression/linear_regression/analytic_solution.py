#  代码实战解析解
#  numpy是用来做数据计算的
#  matplotlib是用来做数据可视化的
import numpy as np
import matplotlib.pyplot as plt

#  有监督的机器学习 
#  生成100个随机数，范围在0-1之间，创建要预测的值的x值
np.random.seed(666)  # 设置随机种子，保证每次运行结果一致
x1 = 2*np.random.rand(100, 1)
x2 = 3*np.random.rand(100, 1)
# 这里是要模拟出来的数据y是代表真实的数据，所以也就是y_hat + error
#  生成100个随机数，范围在0-1之间，加上3x+2,创建要预测的值的y值
y = 4 * x1 + 2 + 3 * x2 + np.random.rand(100, 1)
print(y)
#  在x前面加上一列全为1的列，用于计算截距
x_b = np.c_[np.ones((100, 1)), x1, x2]
# 实现解析解的公式来求解  .T实现矩阵转置
theta = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
print(theta)  # 打印解析解

#  使用模型去做预测
x_new = np.array([[0, 1],
                  [2, 4]])  # 添加两行一列的矩阵
x_new_b = np.c_[np.ones((2, 1)), x_new]  # 添加一列全为1的列，用于计算截距
y_predict = x_new_b.dot(theta)  # 模型预测,这里就是用解析解的公式来计算，x_new_b.dot(theta) = x_new_b * theta
print(y_predict)  # 打印预测结果
#  画图进行展示真实的数据点和我们预测用的数据点
plt.plot(x_new[:, 0], y_predict, "r-"), plt.plot(x1, y, "b.")  # 画图，x_new是横坐标，y_predict是纵坐标，"r-"代表红色实线，"b."代表蓝色点
plt.axis([0, 4, 0, 30])  # 设置x轴和y轴的范围
plt.show()