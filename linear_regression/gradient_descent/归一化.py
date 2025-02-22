from sklearn.preprocessing import StandardScaler
data = [[1], [2], [3], [5], [5]]
scaler = StandardScaler()  # 标准化
print(scaler.fit(data))  # 拟合数据
print(scaler.mean_)  # 输出均值
print(scaler.var_)  # 输出方差
print(scaler.transform(data))  # 转换数据
