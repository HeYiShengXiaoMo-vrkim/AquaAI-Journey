import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(2) #  设置随机种子

# ======================= stap 1/5 生成数据 ======================
sample_nums = 100
mean_value = 1.7
bias = 1
n_data = torch.ones(sample_nums, 2)
x0 = torch.normal(mean_value * n_data, 1) + bias      # 类别0 数据 shape=(100, 2)
y0 = torch.zeros(sample_nums)                         # 类别0 标签 shape=(100, 1)
x1 = torch.normal(-mean_value * n_data, 1) + bias     # 类别1 数据 shape=(100, 2)
y1 = torch.ones(sample_nums)                          # 类别1 标签 shape=(100, 1)
train_x = torch.cat((x0, x1), 0)
train_y = torch.cat((y0, y1), 0).view(-1, 1)

# ======================= step 2/5 选择模型 ======================
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.linear(x)  # 修正：将 features 改为 linear
        x = self.sigmoid(x)
        return x
model = LogisticRegression()  # 选择交叉熵作为损失函数

# ======================= step 3/5 选择损失函数 ======================
criterion = nn.BCELoss()  # 选择随机梯度下降作为优化器

# ======================= step 4/5 选择优化器 ======================
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# ======================= step 5/5 模型训练 ======================
for epoch in range(1000):
    optimizer.zero_grad()  # 新增：梯度清零
    # 前向传播
    y_pred = model(train_x)
    # 计算loss
    loss = criterion(y_pred, train_y)
    # 反向传播
    loss.backward()
    # 更新参数
    optimizer.step()
    print(loss.item())
    if epoch % 10 == 0:
        mask = y_pred.ge(0.5).float()  # 移除squeeze()保持维度一致
        correct = (mask == train_y).sum()
        acc = correct.item() / train_y.size(0)
        print('epoch: ', epoch, 'loss: ', loss.item(), 'acc: ', acc)
        if acc > 0.9:
            break
# 可视化
w0, w1 = model.linear.weight[0]
w0 = float(w0.item())
w1 = float(w1.item())
b = float(model.linear.bias.item())
plot_x = np.arange(-6, 6, 0.1)
plot_y = (-w0 * plot_x - b) / w1
plt.plot(plot_x, plot_y)
x0_data = train_x[np.where(train_y == 0.0)]
x1_data = train_x[np.where(train_y == 1.0)]
plt.scatter(x0_data[:, 0], x0_data[:, 1], c='r', marker='o')
plt.scatter(x1_data[:, 0], x1_data[:, 1], c='b', marker='x')
plt.show()