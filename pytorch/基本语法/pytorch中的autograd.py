from __future__ import print_function
import torch

"""
整个pytorch框架中,所有神经网络本质上都是一个autograd package(自动求导工具包)
autograd package提供了一个对Tensors上所有操作进行自动微分功能
"""
x = torch.ones(2, 2, requires_grad=True)
# 会追踪这个类上定义的所有操作,当代码要进行反向传播时,调用.backward()可计算梯度
# 在这个Tensor上所有的梯度都将被累加到.grad()中

# 梯度计算,跟踪tensor的运算逻辑
a = torch.randn(2, 2)
a = ((a*3)/(a-1))
print(a.requires_grad) # requires_grad是一个布尔值,表示是否需要计算梯度
# requires_grad可以通过with torch.no_grad():在不改变原状态的情况下去除
a.requires_grad_(True) # 设置requires_grad为True
print(a.requires_grad) # requires_grad是一个布尔值,表示是否需要计算梯度
b = (a*a).sum()
print(b.grad_fn) # grad_fn是一个Function,表示这个Tensor是如何计算出来的
# 可以通过.detach()获得一个新的Tensor,这个Tensor和原来的Tensor共享数据,但是不共享梯度
y = x.detach()
print(x.eq(y).all()) # 只比较数字，不比较属性，所以全等

# 反向传播
b.backward() # 反向传播
print(x.grad) # grad是一个Tensor,表示这个Tensor的梯度
