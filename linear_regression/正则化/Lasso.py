"""
Lasso 回归（L1 正则化）

一、核心原理
损失函数 = 均方误差(MSE) + α * ||w||₁
- ||w||₁ 表示权重系数的L1范数（绝对值之和）
- 通过α参数控制正则化强度，实现：
  1. 自动特征选择：将不重要特征的系数压缩至零
  2. 防止过拟合：限制模型复杂度

二、适用场景
 高维数据集（特征数 > 样本数）
 需要特征选择的场景
 存在共线性特征时

三、优缺点
 优势：
  - 生成稀疏解，提升模型可解释性
  - 内置特征选择，避免手动筛选
 限制：
  - 对异常值敏感
  - 当特征高度相关时，可能随机选择其中一个

四、参数说明
- alpha：正则化强度（默认=1.0）
  增大 → 更强正则化，更多系数为零
- max_iter：优化算法迭代次数
  复杂数据集需要增大该值保证收敛
"""

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor

x = 2*np.random.rand(100, 1)
y = 4+3*x+np.random.randn(100, 1)

lasso_reg = Lasso(alpha=0.15, max_iter=10000)
lasso_reg.fit(x, y)
print("系数w1 = ", lasso_reg.coef_)  # 输出回归系数
print("截距w0 = ", lasso_reg.intercept_)  # 输出截距
# 可以看到，Lasso回归将一些回归系数压缩为零，从而实现了特征选择。
# Lasso回归的alpha参数控制了L1范数惩罚项的强度，alpha越大，惩罚越强，回归系数越接近零。
# Lasso回归的max_iter参数控制了最大迭代次数，以防止算法陷入局部最优解。
# Lasso回归是一种有效的特征选择方法，可以用于处理高维数据。
# Lasso回归的缺点是，它可能会产生稀疏解，即一些回归系数为零，这可能会导致模型解释性较差。
# Lasso回归的另一个缺点是，它可能会产生过拟合，特别是在数据量较少的情况下。
print("1的预测值是", lasso_reg.predict([[1]]))  # 输出预测值


# 使用SGDRegressor实现Lasso回归
sgd_reg = SGDRegressor(penalty="l1", alpha=0.15, max_iter=10000)
sgd_reg.fit(x, y.ravel())  # 注意，SGDRegressor的fit方法需要将y转换为1维数组,即使用ravel()方法
print("系数w1 = ", sgd_reg.coef_)  # 输出回归系数
print("截距w0 = ", sgd_reg.intercept_)  # 输出截距
print("1的预测值是", sgd_reg.predict([[1]]))  # 输出预测值