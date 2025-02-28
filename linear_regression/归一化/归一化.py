"""
归一化
一、Z-score 标准化 (Standardization)
公式：z = (x - μ) / σ
- μ: 特征列的均值，σ: 特征列的标准差
- 使数据服从均值为0、标准差为1的分布
- 适用于：数据分布近似正态、需要保留异常值信息的情况
- 特点：保留原始分布形态，对异常值敏感

二、Min-Max 归一化 (Normalization)
公式：x' = (x - x_min) / (x_max - x_min)
- x_min: 特征列最小值，x_max: 特征列最大值
- 将数据压缩到[0,1]区间
- 适用于：数据边界明确、需要快速比较相似性的场景
- 特点：对异常值非常敏感，会改变数据分布

三、方法选择建议：
- 本文件示例使用Z-score标准化（见下方StandardScaler）
- 若数据包含极端值（如5001），建议优先考虑Z-score
- 若需要保证严格数值范围，可考虑RobustScaler（抗异常值版本）
"""

from sklearn.preprocessing import StandardScaler
data = [[1], [2], [3], [5], [5001]]  # 使用了5001这个离群值
scaler = StandardScaler()
print(scaler.fit(data))  # 对数据进行拟合，计算出均值和方差，返回StandardScaler 的一些参数信息
print(scaler.scale_)  # 标准差
print(scaler.var_)  # 方差
print(scaler.transform(data))  # 打印标准化以后的结果
