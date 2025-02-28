"""
原因：1.解决欠拟合问题  2.提高模型准确率  3.可以将原来的线性不可分情况变为线性可分情况
地位：与归一化一同作为数据预处理手段
当y不是随x线性变化时，1. 使用非线性算法(回归树，神经网络). 2.将数据变为线性关系进行计算
线性算法运算速度快是一大优势
目的：自己和自己相乘，组成二阶甚至更高维度
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures  # 多项式回归升维
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error  # 评估升维效果

np.random.seed(100)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)  # 二次项非线性，一次项加倾斜度，常数项移动图像，随机噪声

# 测试集划分，前80个样本用来训练模型，剩下的样本用来测试模型在未知数据上的表现
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='b', alpha=0.5, label='Data points')

d = {1: ('g-', 'Linear'), 2: ('r-', 'Quadratic'), 10: ('y-', '10th degree')}

for degree, (color, label) in d.items():
    # 多项式特征生成器（核心升维组件）
    # degree：指定生成多项式的最高次数，例如degree=2会生成X和X²
    # include_bias=False：不生成全1的偏置项，因为LinearRegression自带截距
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)  # include_bias=False表示不添加常数项，degree表示多项式的次数
    # 训练集升维处理（核心步骤）
    # fit_transform：1. 学习特征组合方式 2. 应用转换到训练数据
    # 生成的特征矩阵包含原始特征及其多项式组合（如X, X², X³...）
    X_poly_train = poly_features.fit_transform(X_train)  # 将训练集数据升维
    # 测试集升维处理（保持与训练集相同的转换方式）
    # 注意：这里使用transform而不是fit_transform，避免数据泄露
    X_poly_test = poly_features.transform(X_test)  # 用与训练数据相同的方式转换测试数据

    # 线性回归模型（在升维后的特征空间进行线性拟合）
    lin_reg = LinearRegression()   # 线性回归
    lin_reg.fit(X_poly_train, y_train)  # 训练升维后的特征矩阵

    X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)  # 生成测试数据
    X_plot_poly = poly_features.transform(X_plot)
    y_plot = lin_reg.predict(X_plot_poly)

    plt.plot(X_plot, y_plot, color, label=f'{label} (degree={degree})')  # 制图，label代表图例

    y_train_pred = lin_reg.predict(X_poly_train)  # 对这些值进行训练
    y_test_pred = lin_reg.predict(X_poly_test)  # 对这些值进行测试，并得到相应的结果

    train_mse = mean_squared_error(y_train, y_train_pred)  # 计算训练集的均方误差
    test_mse = mean_squared_error(y_test, y_test_pred)  # 计算测试集的均方误差

    print(f"Degree {degree}:")
    print(f"  Train MSE: {train_mse:.4f}")
    print(f"  Test MSE: {test_mse:.4f}")  # 输出测试集合的mse，保留四位小数
    print(f"  Coefficients: {lin_reg.coef_}")  # 输出回归系数
    print(f"  Intercept: {lin_reg.intercept_}\n")  # 输出截距

plt.xlabel('X')
plt.ylabel('y')
plt.legend()  # 显示图例
plt.title('polynomial_regression')  # 给出标题
plt.show()