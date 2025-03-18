from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

"""
clf = MLPClassifier(
    solver='sgd',      # 优化算法：随机梯度下降（可选 'adam'/'lbfgs'）
    alpha=1e-5,        # L2正则化系数，防止过拟合（值越小正则化越弱）
    activation='relu', # 激活函数：ReLU（比sigmoid/tanh更高效）
    hidden_layer_sizes=(5, 2),  # 隐藏层结构：两层（5神经元→2神经元）
    max_iter=2000,     # 最大迭代次数（训练轮数上限）
    tol=1e-4,          # 损失下降阈值（连续两次损失差<1e-4则提前停止
    # 其他重要参数（当前未设置）：
    # learning_rate_init=0.001,  # 学习率（当solver='sgd'时必须设置）
    # random_state=42,           # 随机种子（保证结果可复现）
    # batch_size='auto'          # 批处理大小（默认200样本/批）
)
"""

x = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = MLPClassifier(solver='sgd', alpha=1e-5, activation='relu', hidden_layer_sizes=(5, 2), max_iter=2000, tol=1e-4)
clf.fit(x, y) # 多层感知机模型训练模型
predicted_value = clf.predict([[2, 2], [-1, -2]])
print(predicted_value) # 多层感知机模型预测
predicted_prob = clf.predict_proba([[2, 2], [-1, -2]])
print(predicted_prob) # 多层感知机模型预测概率
print([coef.shape for coef in clf.coefs_])
print([coef for coef in clf.coefs_])