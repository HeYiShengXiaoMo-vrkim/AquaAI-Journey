# 正则化，就是鲁棒性调优，牺牲一部分准确性换取模型的泛化能力
# L2正则 ＋ 多元线性回归 == Ridge回归
# L2正则化：在损失函数中加上参数的平方和，使得参数尽量小，防止过拟合
# 线性回归的多元回归对象是mse，Ridge回归的多元回归对象是mse + L2正则化
# L2正则化的结果通常是使各个w参数变小
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor

x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)
ridge_reg = Ridge(alpha=0.1, solver="sag")  # alpha是正则化系数，solver是求解方法,sag是随机平均梯度下降
ridge_reg.fit(x, y)  # 根据给出的数据进行Ridge模型训练，并给出最优的模型参数
print("1.5的预测值是", ridge_reg.predict([[1.5]]))  # 使用训练好的模型对输入值1.5进行预测
print("截距w0 =", ridge_reg.intercept_)  # 输出截距
print("系数w1 =", ridge_reg.coef_)  # 输出系数，正则化可能会影响系数的大小

# 使用SGDRegresser实现Ridge回归
sgd_reg = SGDRegressor(penalty="l2", alpha=0.1)  # penalty是正则化方法，alpha是正则化系数
sgd_reg.fit(x, y.ravel())  # 根据给出的数据进行Ridge模型训练，并给出最优的模型参数
print("1.5的预测值是", sgd_reg.predict([[1.5]]))  # 使用训练好的模型对输入值1.5进行预测
print("截距w0 =", sgd_reg.intercept_)  # 输出截距
print("系数w1 =", sgd_reg.coef_)  # 输出系数，正则化可能会影响系数的大小
# 正则化系数alpha越大，模型越简单，越不容易过拟合，但过小会导致欠拟合
# 正则化系数alpha越小，模型越复杂，越容易过拟合，但过大会导致欠拟合