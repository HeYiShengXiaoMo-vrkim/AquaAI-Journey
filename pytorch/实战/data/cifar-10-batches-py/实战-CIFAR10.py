import torch
import torchvision
import torchvision.transforms as transforms

# 一. 数据集下载
# 定义数据预处理管道
transform = transforms.Compose([
    transforms.ToTensor(),          # 将PIL图像转换为Tensor格式（范围[0,1]）
    transforms.Normalize(           # 标准化处理（范围变为[-1,1]）
        (0.5, 0.5, 0.5),           # RGB三个通道的均值
        (0.5, 0.5, 0.5))           # RGB三个通道的标准差
])

# 创建CIFAR10训练集
train_set = torchvision.datasets.CIFAR10(
    root='./data',                  # 数据集存储路径
    train=True,                     # 加载训练集
    download=True,                 # 如果本地不存在则自动下载
    transform=transform)           # 应用定义的数据预处理

# 创建训练集数据加载器
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=4,                  # 每个batch包含4个样本
    shuffle=True,                  # 每个epoch打乱数据顺序（防止模型记忆顺序）
    num_workers=2)                 # 使用2个子进程加载数据

# 创建CIFAR10测试集
test_set = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,                   # 加载测试集
    download=True,
    transform=transform)

# 创建测试集数据加载器
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=4,
    shuffle=False,                 # 测试集不需要打乱顺序
    num_workers=2)
# 数据集的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 二. 图片展示
# 导入画图包和numpy（将导入移到主函数内部）
import matplotlib.pyplot as plt
import numpy as np

# 构建展示图片的函数
def imshow(img):
    # 对图片进行归一化处理
    img = img / 2 + 0.5
    # 转化为numpy数组
    npimg = img.numpy()
    # 展示图片
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('eaxmple.png')
    plt.show()

"""
# 从数据迭代器中读取一张图片（修改next()调用方式）
dataiter = iter(train_loader)
images, labels = next(dataiter)  # 使用Python内置next函数
# 展示图片
imshow(torchvision.utils.make_grid(images))
# 打印标签
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
"""

# 三.构建卷积网络
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 第一个卷积层（输入通道3-RGB，输出通道6，卷积核5x5）
        self.conv1 = nn.Conv2d(3, 6, 5)  # 输入尺寸32x32 → 输出尺寸28x28（(32-5)/1 +1）
        # 最大池化层（窗口2x2，步长2）
        self.pool = nn.MaxPool2d(2, 2)   # 尺寸减半：28x28 → 14x14
        # 第二个卷积层（输入通道6，输出通道16，卷积核5x5）
        self.conv2 = nn.Conv2d(6, 16, 5) # 输入14x14 → 输出10x10
        # 全连接层1（输入维度16*5*5=400，输出120）
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5来自池化后的特征图尺寸（10x10池化→5x5）
        # 全连接层2（120 → 84）
        self.fc2 = nn.Linear(120, 84)
        # 输出层（84 → 10类别）
        self.fc3 = nn.Linear(84, 10)     # 十个不同类别，输出对应类别分数

    def forward(self, x):
        # 第一层卷积 → ReLU激活 → 池化
        x = self.pool(F.relu(self.conv1(x)))  # 输出形状: (batch_size, 6, 14, 14)
        # 第二层卷积 → ReLU激活 → 池化
        x = self.pool(F.relu(self.conv2(x)))  # 输出形状: (batch_size, 16, 5, 5)
        # 展平特征图（保留batch维度，合并通道和空间维度）
        x = x.view(-1, 16 * 5 * 5)       # 输出形状: (batch_size, 400)
        # 全连接层1 → ReLU激活
        x = F.relu(self.fc1(x))          # 输出形状: (batch_size, 120)
        # 全连接层2 → ReLU激活
        x = F.relu(self.fc2(x))          # 输出形状: (batch_size, 84)
        # 最终输出层（不做激活，后续配合交叉熵损失函数）
        x = self.fc3(x)                  # 输出形状: (batch_size, 10)
        return x
net = Net()
print('神经网络初始化完成')

# 四.定义损失函数和优化器
import torch.optim as optim

criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # 随机梯度下降优化器

# 五.训练网络
for epoch in range(2):  # 迭代2次整个数据集
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # data中包含输入和标签，分别赋值给inputs和labels
        inputs, labels = data # 获取输入和标签
        # 1. 梯度清零
        optimizer.zero_grad()

        # 2. 前向传播,利用网络的输入inputs和标签labels计算输出
        outputs = net(inputs)

        # 3. 利用网络的输出output和标签labels计算损失
        loss = criterion(outputs, labels)

        # 4. 反向传播+参数更新,是标准代码的标准流程
        loss.backward() # 反向传播
        optimizer.step() # 更新权重

        # 打印轮次和损失值
        running_loss += loss.item() # 累加损失
        if (i + 1) % 2000 == 0: # 每2000个小批量数据打印一次训练状态
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('训练完成')
# 保存模型
path = './cifar_net.pth'
torch.save(net.state_dict(), path)

# 五.测试网络
# 1. 在测试集中取出一个批次的数据，做图像和标签的展示
dataiter = iter(test_loader)
images, labels = next(dataiter)
# 展示图片
imshow(torchvision.utils.make_grid(images))
# 打印图片的真实标签
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
# 2. 加载模型
# 首先实例化模型的类对象
net = Net()
# 加载训练阶段保存好的模型的状态字典
net.load_state_dict(torch.load(path))
# 利用模型对图像进行预测
outputs = net(images)
# 共有十个类别，对每个图像，利用softmax函数计算每个类别的概率
_, predicted = torch.max(outputs, 1)
# 打印标签预测的结果
print('Predicted:', ''.join('%5s' % classes[predicted[j]] for j in range(4)))

# 3. 计算整个测试集的准确率
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct/total))

# 4. 分类别准确度计算
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))