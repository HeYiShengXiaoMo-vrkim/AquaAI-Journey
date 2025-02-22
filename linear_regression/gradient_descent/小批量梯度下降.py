import numpy as np

np.random.seed(42)
x = 2*np.random.rand(300, 1)
y = 4 + 3*x + np.random.randn(300, 1)
x_b = np.c_[np.ones((300, 1)), x]

t0, t1 = 5, 500


# 定义一个函数调整学习率
def learning_rate_schedule(t):
    return t0 / (t + t1)

n_epochs = 100000  # 轮次
m = 300  # 训练样本的总模拟数
batch_size = 10  # 每次更新梯度使用的子集合大小
num_batches = int(m / batch_size)  # 每轮训练中，完整数据集可以被分成的批次数量，共有一百个样本，每次拿十个出来训练，每轮训练十个批次

theta = np.random.randn(2, 1)  # θ 是待优化的参数向量，包含了截距项和斜率项。
for epoch in range(n_epochs):
    arr = np.arange(len(x_b))
    np.random.shuffle(arr)
    x_b = x_b[arr]
    y = y[arr]
    for i in range(num_batches):
        x_batch = x_b[i * batch_size: (i + 1) * batch_size]  # 提取从第 i * batch_size 个样本到第 (i + 1) * batch_size 个样本
        y_batch = y[i * batch_size: (i + 1) * batch_size]
        gradients = x_batch.T.dot(x_batch.dot(theta) - y_batch)
        learning_rate = learning_rate_schedule(epoch * num_batches + i)  # 调整学习率
        theta = theta - learning_rate * gradients

print(theta)
