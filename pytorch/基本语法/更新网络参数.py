# 这里使用随机梯度下降SGD
# weight = weight-learning_rate*gradient

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data*learning_rate)

# 首先导入优化器的包，optim中包含若干常用的优化算法,比如SGD,ADAM
import torch.optim as optim
# 创建一个优化器，SGD只需要传入待优化的参数和学习率
optimizer = optim.SGD(net.parameters(),lr=0.01)
# 接下来，我们只需要在训练过程中，先将梯度清零，然后反向传播计算梯度，最后优化器更新参数
optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
output = net(input)
loss = criterion(output,target)
# 对损失值执行反向传播的操作
loss.backward()
optimizer.step() # 计算完毕不代表参数已经更新，更新参数