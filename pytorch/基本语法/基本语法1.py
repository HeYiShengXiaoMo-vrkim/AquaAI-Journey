"""
pytorch 是基于numpy的科学计算包,作为numpy的替代品,可以利用gpu的性能进行计算
"""

from __future__ import print_function
import torch

# 1. 创建矩阵
x = torch.empty(5, 3) # 未初始化的5x3矩阵 脏数据
x1 = torch.rand(5, 3) # 有初始化矩阵
x2 = torch.zeros(5, 3, dtype=torch.long) # 全0矩阵, 数据类型为long
x3 = torch.tensor([5.5, 3]) # 直接从数据构造张量
x4 = x.new_ones(5, 3, dtype=torch.double) # 从x继承数据类型
x5 = torch.randn_like(x, dtype=torch.float) # 从x继承数据类型, 但是随机初始化

# 2. 运算 - 加法
# 第一种,直接相加
print(x + x1)
# 第二种, 函数相加
print(torch.add(x, x1))
# 第三种, 函数相加, 指定输出
result = torch.empty(5, 3, dtype=torch.double)
torch.add(x, x1, out=result)
print(result)
# 所有in_place操作,函数都有下划线后缀,比如x.copy_(y),x.add_(y)都会改变x的值

# 3. 切片
print(x[:, 1]) # 第二列, 所有行

# 4. 改变形状,使用view要保证元素个数一致
x = torch.randn(4, 4)
y = x.view(16) # 改变形状,一行十六列
z = x.view(-1, 8) # 改变形状, -1表示自动计算维度
print(x.size(), y.size(), z.size())
# 张量只有一个元素,可以使用.item()来获取值
x = torch.randn(1)
print(x)
print(x.item())

# 5. 张量和numpy数组相互转换
# 张量转numpy数组
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
# 转换numpy到张量
# 所有在cpu上的张量除了CharTensor都支持转换为numpy数组并可以进行反向传播, 反之亦然
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a)

# 6. CudaTensor : Tensors可用.to()方法将其移动到任意设备
x = torch.rand(5, 3)
if torch.cuda.is_available():
    # 定义一个设备对象，这里指定为cuda
    device = torch.device("cuda")
    # 直接在gpu创建Tensor
    y = torch.ones_like(x, device=device)
    # cpu上张量移动到gpu
    x = x.to(device)
    z = x + y
    print(z) # z在gpu上
    # 把gpu上的张量移动到cpu
    print(z.to("cpu", torch.double)) # to()还可以同时更改数据类型