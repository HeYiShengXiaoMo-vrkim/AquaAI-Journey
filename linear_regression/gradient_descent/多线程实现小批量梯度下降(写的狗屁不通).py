import threading
import numpy as np

# 全局变量
data_ready = threading.Event()
calculation_done = threading.Event()

# 定义数据
np.random.seed(42)  # 为了可重复性设置随机种子
x = 2 * np.random.rand(300, 1)
y = 3 * x + 4 + np.random.randn(300, 1)
x_b = np.c_[np.ones((300, 1)), x]  # 添加偏置项

# 控制参数
t0, t1 = 5, 500
n_iterations = 100000  # 总迭代次数
m = 300
batch_size = 10
theta = np.random.randn(2, 1)


def learning_rate(t):
    return t0 / (t + t1)


def data_preparation():
    global x_b, y
    for _ in range(n_iterations):
        arr = np.arange(m)
        np.random.shuffle(arr)
        x_b = x_b[arr]
        y = y[arr]
        data_ready.set()  # 数据已准备好
        calculation_done.wait()  # 等待计算完成
        calculation_done.clear()  # 重置事件


def gradient_descent():
    global theta
    for i in range(n_iterations):
        data_ready.wait()  # 等待数据准备好
        data_ready.clear()  # 重置事件

        x_batch = x_b[0:batch_size]
        y_batch = y[0:batch_size]
        gradients = 2 / batch_size * x_batch.T.dot(x_batch.dot(theta) - y_batch)
        lr = learning_rate(i)
        theta = theta - lr * gradients

        calculation_done.set()  # 计算完成


def main():
    thread1 = threading.Thread(target=data_preparation)
    thread2 = threading.Thread(target=gradient_descent)

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    print("最终 theta:", theta)
    print("预测值 (3, 4):", theta[0] + 3 * theta[1])


if __name__ == '__main__':
    main()