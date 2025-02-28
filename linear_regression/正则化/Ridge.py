"""
Ridge 回归（L2 正则化）

一、核心原理
损失函数 = 均方误差(MSE) + α * ½||w||₂²
- ||w||₂² 表示权重系数的L2范数平方（平方和）
- 通过α参数控制正则化强度，实现：
  1. 约束参数幅度，防止过拟合
  2. 改善条件数，提升数值稳定性

二、适用场景
 特征高度相关（多重共线性）
 需要平滑系数变化的场景
 数据维度 < 样本数量时

三、优缺点
 优势：
  - 数学闭式解确保收敛
  - 对异常值相对鲁棒
 限制：
  - 不产生稀疏解
  - 特征选择需配合其他方法

四、参数说明
- alpha：正则化强度（默认=1.0）
  增大 → 更强约束，系数更趋近零
- solver：优化算法
  'sag'：适合大数据集，'svd'：适合特征数较多
"""

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