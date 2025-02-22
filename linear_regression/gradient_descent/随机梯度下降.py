import numpy as np
# 引入了随机性，避免掉进局部最优解，同时使用单一样本的优化，实现了速度上的优化

# 创建目标变量
x = 2 * np.random.rand(100, 1)
y = 4 + 3*x + np.random.randn(100, 1)
x_b = np.c_[np.ones((100, 1)), x]

# 控制学习率变化
t0, t1 = 5, 500

# 定义函数控制学习率
def learning_schedule(t):
    return t0 / (t + t1)

# 定义执行的轮次
n_epochs = 1000

# 定义每轮的批次
m = 100

# theta初始化，一开始是随机值，最后进行逼近
theta = np.random.randn(2, 1)

# 大轮次做主循环，批次作为副循环
for epoch in range(n_epochs):
    arr = np.arange(len(x_b))  # 创建一个范围从0~len-1的序列
    np.random.shuffle(arr)  # 打乱数据，避免导致序列顺序依赖，掉入局部最优解
    x_b = x_b[arr]  # 更新x_b数组的内容
    y = y[arr]  # 使用x的数据更新y,保持x_b和y的对应关系
    for i in range(m):
        xi = x_b[i:i+1]  # 取第i个样本
        yi = y[i:i+1]  # 提取单独样本是随机梯度下降的核心所在，可以对局部变量进行优化
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)  # 计算梯度
        eta = learning_schedule(epoch * m + i)  # 计算学习率
        theta = theta - eta * gradients  # 更新theta

print(theta)