"""
归一化
当各个影响因子的数量级不一致时，会导致不同因素到达目标值的速度不一致
本质是统一步调，将数量级进行统一，标准化公式是 z =  (原始值 - 均值) / 标准差
矩阵运算中使用x_ij = (x_ij - x_j^min) / (x_j^max - x_j^min)
减去平均值，使得一部分元素在0(某个分界线)以上，另一部分元素在0(某个分界线)以下，便于对元素进行划分
除以方差，使得元素受离群值的影响减小
"""

from sklearn.preprocessing import StandardScaler
data = [[1], [2], [3], [5], [5001]]  # 使用了5001这个离群值
scaler = StandardScaler()
print(scaler.fit(data))  # 对数据进行拟合，计算出均值和方差，返回StandardScaler 的一些参数信息
print(scaler.scale_)  # 标准差
print(scaler.var_)  # 方差
print(scaler.transform(data))  # 打印标准化以后的结果