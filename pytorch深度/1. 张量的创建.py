import torch
import numpy as np

"""
1. 第一种创建方式: 直接创建
torch.tensor(
    data,   数据，可以是列表，元组，numpy数组，标量
    dtype=torch.float32,  数据类型
    device='cpu',   设备
    requires_grad=False,   是否需要计算梯度
    pin_memory=False  是否需要锁页内存
)
"""
arr = np.ones((3, 3))
print(arr)
t = torch.tensor(
    arr,
    dtype=torch.float32,
    device='cpu',
    requires_grad=False,
    pin_memory=False
)

"""
2. 第二种创建方式: 从numpy中创建
torch.from_numpy(ndarray)
从numpy中创建的tensor与numpy共享内存，当修改其中一个数据，另外一个也会被修改
"""
t = torch.from_numpy(arr)
print(t)

"""
3. 依据数值创建
3.1 依据size创建全0张量
torch.zeros(
    *size,   形状
    out=None,   输出的张量
    dtype=torch.float32,  数据类型
    layout=torch.strided,  内存中的布局形式,有strided, sparse_coo
    device='cpu',   设备
    requires_grad=False,   是否需要计算梯度
    )
3.2 依据input形状创建全0张量
torch.zeros_like(
    input,   创建与input形状相同的全0张量
    dtype=None,   数据类型
    layout=None,   内存中的布局形式,有strided, sparse_coo
    device=None,   设备
    requires_grad=False,   是否需要计算梯度
)
3.3 依据size创建全1张量
torch.ones()
torch.ones_like() 同上
3.4 依据size创建随机数张量
torch.rand()
torch.rand_like() 同上
3.5 根据size创建指定数值的张量
torch.full(
    size,   形状
    fill_value,   填充的值
    out=None,   输出的张量
    dtype=None,   数据类型
    device=None,   设备
    requires_grad=False,   是否需要计算梯度
    layout=torch.strided,  内存中的布局形式,有strided, sparse_coo
)
torch.full_like() 同上
3.6 依据数值区间和步长创建张量
torch.arange(
    start=0,   起始值
    end,   结束值
    step=1,   步长
    out=None,   输出的张量
    dtype=None,   数据类型
    layout=torch.strided,  内存中的布局形式,有strided, sparse_coo
    device=None,   设备
    requires_grad=False,   是否需要计算
)
3.7 创建均分的1维张量
torch.linspace(
    start,   起始值
    end,   结束值
    steps=100,   数列的长度
    out=None,   输出的张量
    dtype=None,   数据类型
    layout=torch.strided,  内存中的布局形式,有strided, sparse_coo
    device=None,   设备
    requires_grad=False,   是否需要计算梯度
)
3.8 创建对数均分的1维张量
torch.logspace(
    start,   起始值
    end,   结束值
    steps=100,   数列的长度
    base=10.0,   对数函数的底数,默认以10为底
    out=None,   输出的张量
    dtype=None,   数据类型
    layout=torch.strided,  内存中的布局形式,有strided, sparse_coo
    device=None,   设备
    requires_grad=False,   是否需要计算梯度
)
3.9 创建对角矩阵
torch.eye(
    n,   矩阵的行数
    m=None,   矩阵的列数
    out=None,   输出的张量
    dtype=None,   数据类型
    layout=torch.strided,  内存中的布局形式,有strided, sparse_coo
    device=None,   设备
    requires_grad=False,   是否需要计算梯度
)

"""
# 3.1
out_t = torch.tensor([1])
torch.ones(
    (3,3),
    dtype=torch.long,
    out=out_t
)
# 3.4
torch.full(
    (3,4),10
)
# 3.6
t2 = torch.arange(
    0,10,1
) # 从0到10，不包括10，步长为1
# 3.7
t3 = torch.linspace(
    0,10,10
)

"""
4.1 高斯分布
torch.normal(
    mean,   均值
    std,   标准差  -> 四种模式,mean为标量，std为标量，mean为标量，std为张量，mean为张量，std为标量，mean为张量，std为张量
    out=None,   输出的张量
    dtype=None,   数据类型
    layout=torch.strided,  内存中的布局形式,有strided, sparse_coo
    device=None,   设备
    requires_grad=False,   是否需要计算梯度
)
4.2 标准正态分布
torch.randn(
    *size,   形状
    out=None,   输出的张量
    dtype=None,   数据类型
    layout=torch.strided,  内存中的布局形式,有strided, sparse_coo
    device=None,   设备
    requires_grad=False,   是否需要计算梯度
    )
torch.randn_like()
4.3 生成均匀分布
torch.randint(
    low=0,   最小值
    high,   最大值
    size,   形状
    out=None,   输出的张量
    dtype=None,   数据类型
    layout=torch.strided,  内存中的布局形式,有strided, sparse_coo
    device=None,   设备
    requires_grad=False,   是否需要计算梯度
)
torch.randn_like()
4.4 生成从0到n-1的随机排列
torch.randperm(
    n,   张量的长度
    out=None,   输出的张量
    dtype=torch.int64,   数据类型
    layout=torch.strided,  内存中的布局形式,有strided, sparse_coo
    device=None,   设备
    requires_grad=False,   是否需要计算梯度
)
4.5 以input为概率,生成伯努利分布(0-1分布,两点分布)
torch.bernoulli(
    input,   概率值
    *, 
    generator=None,
    out=None
)
"""
# 4.1
# 双张量
# mean = torch.arange(1,5,dtype=torch.float32)
# std = torch.arange(1,5,dtype=torch.float32)
# t_normal = torch.normal(mean,std) # 每个元素都有一个对应的均值和标准差
# print(t_normal)
# 张量加标量
# mean = torch.arange(1,5,dtype=torch.float32)
# std = 1
# t_normal = torch.normal(mean,std) # 每个元素都有一个对应的均值和标准差
# print(t_normal)
# 标量加张量
# mean = 1
# std = torch.arange(1,5,dtype=torch.float32)
# t_normal = torch.normal(mean,std) # 每个元素都有一个对应的均值和标准差
# print(t_normal)
# 双标量
# mean = 1
# std = 2
# t_normal = torch.normal(mean, std)