# 导入所需库
from sklearn.datasets import load_iris  # 加载鸢尾花数据集
import numpy as np  # 数值计算库
from sklearn.linear_model import LogisticRegression  # 逻辑回归模型

# 加载并探索数据集
iris = load_iris()
list(iris.keys())  # 将数据集键转换为列表
print(iris['DESCR'])  # 打印数据集描述
print(iris['feature_names'])  # 打印特征名称

# 准备数据
x = iris['data'][:, 3:]  # 只使用第四个特征（花瓣宽度）
print(x)
y = (iris['target'] == 2).astype(np.int)  # 创建二分类标签（是否是山鸢尾）
print(y)

# 创建并训练模型
binary_classifier = LogisticRegression(solver='sag', max_iter=1000)  # 使用随机平均梯度下降求解器
binary_classifier.fit(x, y)  # 训练模型

# 生成预测数据
x_new = np.linspace(0, 3, 1000).reshape(-1, 1)  # 创建0-3范围内的测试数据
print(x_new)
y_proba = binary_classifier.predict_proba(x_new)  # 预测概率值
print(y_proba)  # 输出类别概率（非山鸢尾 vs 山鸢尾）
y_hat = binary_classifier.predict(x_new)
print(y_hat) # 输出预测类别


## 实现多分类
## log_reg = LogisticRegression(C=1000, multi_class='ovr', solver='sag', max_iter=1000)