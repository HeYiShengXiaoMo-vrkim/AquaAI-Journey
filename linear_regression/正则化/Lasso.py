# Lasso:L1正则化
# Lasso回归是一种使用L1正则化的线性回归模型，它通过在损失函数中添加L1范数惩罚项来防止过拟合。L1范数惩罚项使得一些回归系数变为零，从而实现特征选择。
# Lasso的副产物是帮助模型进行特征筛选

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