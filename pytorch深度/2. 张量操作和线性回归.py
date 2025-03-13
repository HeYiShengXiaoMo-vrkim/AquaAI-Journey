"""
一、张量操作
1.1 张量的拼接
~   torch.cat(tensors, dim=0, out=None)
    功能：将张量按维度dim进行拼接,不会扩张张量的维度
    参数：
        tensors：张量序列
        dim：要拼接的维度
        out：输出的张量
    注意：
        张量序列中的张量除了拼接的维度外，其他维度必须相同
~   torch.stack(tensors, dim=0, out=None)
    功能：在新创建的维度dim上进行拼接,会扩张张量的维度
    参数：
        tensors：张量序列
        dim：要拼接的维度
        out：输出的张量
    注意：
        张量序列中的张量必须完全相同
1.2 张量的切分
~   torch.chunk(input, chunks, dim=0)
    功能：将张量按维度dim进行平均切分,返回值为张量序列
    参数：
        input：要切分的张量
        chunks：要切分的份数
        dim：要切分的维度
    注意：
        若不能整除，最后一份张量小于其他张量
~   torch.split(tensor, split_size_or_sections, dim=0)
    功能：将张量按维度dim进行切分,返回值为张量序列
    参数：
        tensor：要切分的张量
        split_size_or_sections：为int时，表示每一份的长度;为list时，表示每一份的长度
        dim：要切分的维度
    注意：
        当split_size_or_sections为int时，若不能整除，最后一份张量小于其他张量
        当split_size_or_sections为list时，list的和必须等于dim维度的长度
"""
import torch
from matplotlib import pyplot as plt

"""
1. 张量的拼接
t = torch.ones((2,3))
t_0 = torch.cat([t,t],dim=0) # 得到一个4*3的张量
t_1 = torch.cat([t,t],dim=1) # 得到一个2*6的张量
print(t_0)
t_2 = torch.stack([t,t],dim=2) # 得到一个2*3*2的张量
print(t_2)
"""

"""
2. 张量的切分
a = torch.ones((2,7))
list_of_tensors = torch.chunk(a,3,dim=1) # 被分为3,3,1
a = torch.ones((2,5))
list_of_tensors = torch.split(a,[2,1,2],dim=1) # 被分为2行2列,1行1列,2行2列
for idx, t in enumerate(list_of_tensors):
    print("第{}个张量：{}, shape is {}".format(idx+1, t, t.shape))

"""
"""
3. 张量的索引
~   torch.index_select(input, dim, index, out=None)
        功能：在维度dim上，按index索引数据
        参数：
            input：要索引的张量
            dim：要索引的维度
            index：要索引数据的序号
        注意：
            index为张量，index中数据必须为long型
    
~   torch.masked_select(input, mask, out=None)
        功能：按mask中的True进行索引
        参数：
            input：要索引的张量
            mask：与input同形状的布尔类型张量
        注意：
            最终的索引结果是一维的
            
t = torch.randint(0,9,size=(3,3))
idx = torch.tensor([0,2],dtype=torch.long) # 注意数据类型为long
t_select = torch.index_select(t,dim=0,index=idx) # 返回0,2行拼接后的张量
print("t:\n{}\nt_select:\n{}".format(t,t_select))


"""
# t = torch.randint(0,9,size=(3,3))
# mask = t.ge(5) # ge is mean greater than or equal/返回一个布尔类型的张量
# t_select = torch.masked_select(t,mask) # 返回大于等于5的张量
# print("t:\n{}\nmask:\n{}\nt_select:\n{} ".format(t,mask,t_select))
"""
3.1 张量的变换
~   torch.reshape(input, shape)
        功能：变换张量的形状
        参数：
            input：要变换的张量
            shape：新的形状
        注意：
            当张量在内存中是连续的，新张量与input共享数据内存
~   torch.transpose(input, dim0, dim1)
        功能：交换张量的两个维度
        参数：
            input：要变换的张量
            dim0：要交换的维度
            dim1：要交换的维度
        注意：
            当dim0和dim1相等时，相当于转置
~   torch.t(input)
        功能：二维张量转置,对矩阵而言，等价于torch.transpose(input, 0, 1)
        参数：
            input：要变换的张量
        注意：
            相当于torch.transpose(input, 0, 1)
~   torch.squeeze(input, dim=None, out=None)
        功能：压缩长度为1的维度（轴）
        参数：
            input：要操作的张量
            dim：若为None，移除所有长度为1的轴；若指定维度，当且仅当该轴长度为1时，可以被移除
        注意：
            返回的张量与input共享内存，除非设置out
~   torch.unsqueeze(input, dim, out=None)
        功能：依据dim扩展维度
        参数：
            input：要操作的张量
            dim：扩展的维度
        注意：
            返回的张量与input共享内存，除非设置out
~   torch.permute(input, dims)
        功能：交换张量的维度
        参数：
            input：要操作的张量
            dims：要交换的维度
        注意：
            返回的张量与input共享内存，除非设置out
"""
# t = torch.randperm(8) # 随机生成0-7的整数
# t_reshape = torch.reshape(t,(-1,2,2)) # -1表示自动计算维度
# t_transpose = torch.transpose(t_reshape,1,2) # 交换维度1和维度2
# print("t:{}\nt_reshape:\n{}\nt_transpose:\n{}".format(t,t_reshape,t))
# t = torch.rand(2, 1, 2, 1)
# t_sq = torch.squeeze(t)
# t_0 = torch.squeeze(t, dim=0)
# t_1 = torch.squeeze(t, dim=1) # 当且仅当该轴长度为1时，可以被移除
# print(t_0.shape)
# print(t_1.shape)
# print(t_sq.shape)
# print(t.shape)

"""
import torch
1. 加法
~   torch.add(input, alpha=1, other, out=None)
     功能：逐元素计算input+alpha*other
     参数：
         input：第一个张量
         alpha：乘项因子
         other：第二个张量
         out：输出的张量
     注意：
         input和other必须形状相同
~   torch.addcdiv(input, value=1, tensor1, tensor2, out=None)
     功能：逐元素计算input+value*tensor1/tensor2
     参数：
         input：第一个张量
         value：乘项因子
         tensor1：第二个张量
         tensor2：第三个张量
         out：输出的张量
     注意：
         input和tensor1,tensor2必须形状相同
~   torch.addcmul(input, value=1, tensor1, tensor2, out=None)
     功能：逐元素计算input+value*tensor1*tensor2
     参数：
         input：第一个张量
         value：乘项因子
         tensor1：第二个张量
         tensor2：第三个张量
         out：输出的张量
     注意：
         input和tensor1,tensor2必须形状相同
"""
import matplotlib.pyplot as plt
lr = 0.1

# 创建训练数据
x = torch.randn(20,1)*10 # x data (tensor), shape=(20, 1)
y = 2*x + (5 + torch.randn(20,1)) # y data (tensor), shape=(20, 1)

# 构建线性回归函数
w = torch.randn((1),requires_grad=True) # 随机初始化参数
b = torch.zeros((1),requires_grad=True) # 随机初始化参数
for iteration in range(1000):
    # 前向传播
    wx = torch.mul(w,x) # w*x
    y_pred = torch.add(wx,b) # w*x+b
    # 计算MSE loss
    loss = (0.5*(y-y_pred)**2).mean()
    # 反向传播
    loss.backward()
    # 更新参数前清空梯度
    if w.grad is not None:
        w.grad.data.zero_()
    if b.grad is not None:
        b.grad.data.zero_()
    # 更新参数
    b.data.sub_(lr*b.grad)
    w.data.sub_(lr*w.grad)
    if iteration % 20 == 0:
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),y_pred.data.numpy(),'r-',lw=5)
        plt.text(2,20,'Loss=%.4f' % loss.data.numpy(),fontdict={'size':20,'color':'red'})
        plt.xlim(1.5,10)
        plt.ylim(8,20)
        plt.title("Iteration: {}\nw: {} b: {}".format(iteration,w.data.numpy(),b.data.numpy()))
        plt.show()
        plt.pause(0.5)

        if loss.data.numpy() < 1:
            break