from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale

data = load_breast_cancer()
x, y = data['data'][:, :2], data['target'] # load_breast_cancer中的数据是字典类型,可以用key索引value,只取两列数据方便进行画图(不至于太乱)

# 求出两个维度对应的数据在逻辑回归中的最优解
lr = LogisticRegression(fit_intercept=False) # 这里的fit_intercept标志着要不要加上w0
lr.fit(x, y)

# 提取出两个维度的w1, w2
theta1 = lr.coef_[0, 0] # w是列向量
theta2 = lr.coef_[0, 1] # 这里分别取出 w1 和 w2

# 实现对应的sigmod函数定义,已知w1,w2,x,返回数据的y_predict
"""
    y_hat = sigmod(z)  
    z=theta.T*x=w1*x1+w2*x2
"""
def p_theta_function(features, w1, w2):
    z = w1*features[0] + w2*features[1]
    return 1 / (1 + np.exp(-z))

# 定义一份已知数据的x,y,如果已知w1,w2的情况下,计算这份数据的loss损失
def loss_function(sample_features, sample_labels, w1, w2):
    result = 0
    # 遍历数据中的每一条样本,并且计算每条样本的损失,加到result身上得到整体的数据集损失
    for features,label in zip(sample_features, sample_labels):
        # 这是计算一条样本中的y_predict
        p_result = p_theta_function(features, w1, w2) # 这里得到sigmod函数的结果
        loss_result = -1*label*np.log(p_result) - (1-label)*np.log(1-p_result) # L = -y*log(p) - (1-y)*log(1-p)
        result += loss_result
    return result

theta1_space = np.linspace(theta1-0.6, theta1+0.6, num=100)
theta2_space = np.linspace(theta2-0.6, theta2+0.6, num=100)

result1_ = np.array([loss_function(x, y, i, theta2) for i in theta1_space])
result2_ = np.array([loss_function(x, y, theta1, i) for i in theta2_space])

fig1 = plt.figure(figsize=(8, 8))
plt.subplot(2,2,1)
plt.plot(theta1_space, result1_, 'b-', label='theta1')

plt.subplot(2,2,2)
plt.plot(theta2_space, result2_, 'g-', label='theta2')

plt.subplot(2,2,3)
theta1_grid, theta2_grid = np.meshgrid(theta1_space, theta2_space)
loss_grid = loss_function(x, y, theta1_grid, theta2_grid)
plt.contour(theta1_grid, theta2_grid, loss_grid, colors='b')

plt.subplot(2,2,4)
plt.contour(theta1_grid, theta2_grid, loss_grid, colors='b')
plt.xlabel('theta1')
plt.ylabel('theta2')
plt.savefig('loss_theta.png')

fig2 = plt.figure(figsize=(10,10))
ax = Axes3D(fig2)
surf = ax.plot_surface(theta1_grid, theta2_grid, loss_grid, cmap='rainbow', rstride=1, cstride=1)
ax.view_init(elev=30, azim=-60)  # 设置俯视角度
fig2.colorbar(surf)  # 添加颜色条
ax.set_xlabel('theta1')
ax.set_ylabel('theta2')
ax.set_zlabel('Loss')
plt.savefig('loss.png')

plt.show()