# 线性回归中，我们使用均方误差(MSE)作为损失函数：
# J(θ) = (1/2m) * Σ(h_θ(x_i) - y_i)²
# 其中 h_θ(x_i) 是模型预测，y_i 是实际值。
# 梯度是损失函数对θ的偏导数。
# 对于线性回归，这个梯度可以表示为：
# ∇J(θ) = (1/m) * X_b.T * (X_b.dot(θ) - y)

import numpy as np

#  创建数据集x, y
np.random.seed(1)  # 为了结果的可复现性
x = 2 * np.random.rand(100, 1)  # 生成100个随机数，范围在0-2之间,x是输入特征，一列一百行
y = 4 + 3 * x + np.random.randn(100, 1)  # y是输出特征，目标变量
X_b = np.c_[np.ones((100, 1)), x]  # 在x前面添加一列全为1的列，用于计算截距项

#  创建超参数
t0, t1 = 5, 500  # t0是学习率，t1是学习率衰减因子
n_iterations = 10000  # 迭代次数

#  定义一个函数来调整学习率,t是迭代次数，相当于说学习率随循环次数递减，避免发生震荡，开始时快速学习，后来精细调整
def learning_schedule(t):
    return t0 / (t + t1)

#  1. 初始化theta, w0...wn, 正态分布创建W
theta = np.random.randn(2, 1)

#  4. 判断是否收敛，一般不会去设置阈值，而是直接采用设置相对大的迭代次数保证可以收敛
for i in range(n_iterations):
    #  2. 计算梯度
    gradients = X_b.T.dot(X_b.dot(theta) - y)
    if gradients.all() == 0:
        break
    # 3,应用梯度下降法的公式去调整theta值 θt+1=θt-η*gradient
    learning_rate = learning_schedule(i)
    theta = theta - learning_rate * gradients

print(theta)