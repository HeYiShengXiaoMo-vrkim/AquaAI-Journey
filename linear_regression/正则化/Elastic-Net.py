"""
Elastic Net 回归（L1+L2 混合正则化）

一、核心原理
损失函数 = 均方误差(MSE) + α * (l1_ratio·||w||₁ + 0.5·(1-l1_ratio)·||w||₂²)
- 平衡 L1 正则化的特征选择能力与 L2 正则化的稳定性
- l1_ratio=1 时退化为 Lasso，l1_ratio=0 时退化为 Ridge

二、适用场景
 特征数量远大于样本数量 (n_features >> n_samples)
 存在高度相关特征组
 需要兼顾特征选择与模型稳定性

三、参数说明
- alpha：总正则化强度（默认=1.0）
  增大 → 更强约束，系数更趋近零
- l1_ratio：L1 正则化占比（0-1）
  0.5 → L1/L2 各占50%
- max_iter：优化算法迭代次数
  复杂数据需要增大该值保证收敛
"""

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor

np.random.seed(1)
x = 6*np.random.rand(100,1)
y = 4+3*x+np.random.randn(100,1)

elastic_net = ElasticNet(alpha=0.4, l1_ratio=0.5) # 正则化强度为0.4，L1正则化占比为0.5
elastic_net.fit(x, y)
print("系数w1 = ", elastic_net.coef_)
print("截距w0 = ", elastic_net.intercept_)
print("训练后6的预测值是：", elastic_net.predict([[6]]))

sgd_reg = SGDRegressor(penalty='elasticnet', alpha=0.4, l1_ratio=0.5, max_iter=1000)
sgd_reg.fit(x, y.ravel())
print("系数w1 = ", sgd_reg.coef_)
print("截距w0 = ", sgd_reg.intercept_)
print("训练后6的预测值是：", sgd_reg.predict([[6]]))