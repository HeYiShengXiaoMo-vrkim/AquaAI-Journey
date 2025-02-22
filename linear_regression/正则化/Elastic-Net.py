import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor

np.random.seed(1)
x = 6*np.random.rand(100,1)
y = 4+3*x+np.random.randn(100,1)

elastic_net = ElasticNet(alpha=0.4, l1_ratio=0.5)
elastic_net.fit(x, y)
print("系数w1 = ", elastic_net.coef_)
print("截距w0 = ", elastic_net.intercept_)
print("训练后6的预测值是：", elastic_net.predict([[6]]))

sgd_reg = SGDRegressor(penalty='elasticnet', alpha=0.4, l1_ratio=0.5, max_iter=1000)
sgd_reg.fit(x, y.ravel())
print("系数w1 = ", sgd_reg.coef_)
print("截距w0 = ", sgd_reg.intercept_)
print("训练后6的预测值是：", sgd_reg.predict([[6]]))