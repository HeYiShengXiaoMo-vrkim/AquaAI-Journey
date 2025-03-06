"""
1. 关于torch.nn
- 使用pytorch构建神经网络,主要工具在torch.nn中
- nn依赖于autograd来定义模型和微分,并对其自动求导
2. 构建神经网络典型流程(参考)
- 定义一个包含可训练参数的神经网络
- 遍历训练数据集
- 处理输入数据，使其流经神经网络
- 计算损失(loss)
- 将网络参数的梯度进行反向传播
- 以一定的规则更新网络的权重
"""
# 3. pytorch实现神经网络
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个简单的网络类
class Net(nn.Module):
    def __init__(self):
        super(Net,  self).__init__() #写死的，照抄就行
        # 定义第一层卷积网络,输入通道维度=1,输出通道维度=10,卷积核大小=5
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # 定义第二层卷积网络,输入通道维度=10,输出通道维度=20,卷积核大小=5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 定义三层全连接网络,输入维度=320,输出维度=50
        self.fc1 = nn.Linear(16*6*6, 120) # 这里的16是根据输入数据的维度而定的
        self.fc2 = nn.Linear(120, 84) # fc2是第一层全连接网络的输出维度,fc3是第二层全连接网络的输入维度
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        # 在(2,2)位置上进行池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 经过卷积层处理，张量要进入全连接层，进入前要调整张量形状
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        # 计算size,除了第0个维度上的batch_size
        size = x.size()[1:] # 这里的1是batch_size,size是一个元组
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

# 1. 模型的所有可训练参数，可以通过net.parameters()返回
params = list(net.parameters())
print(len(params))
print(params[0].size())
# 2. 随机生成一个输入，32是batch_size,1是通道数,32是图片大小
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
# 3. 有了输出张量以后,执行梯度归零和反向传播操作
net.zero_grad()
out.backward(torch.randn(1, 10))
# 4. torch构建的神经网络只支持mini-batch的方式,不支持一次只输入一个样本,
# 比如,nn.Conv2d()只支持4维的输入，即nSamples x nChannels x Height x Width
# 如果是一个样本,则应该使用input.unsqueeze(0)来添加一个假的batch_size维度,主动将3DTensor扩充为4DTensor

