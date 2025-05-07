import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 解决中文乱码问题
rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei
rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 定义原函数
def f(x):
    return 1 / (1 + 25 * x**2)

# 拉格朗日插值多项式
def lagrange_interpolation(x, x_nodes, y_nodes):
    n = len(x_nodes)
    L = np.zeros_like(x)
    for i in range(n):
        li = np.ones_like(x)
        for j in range(n):
            if i != j:
                li *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        L += y_nodes[i] * li
    return L

# 生成均匀分点和切比雪夫节点
def uniform_nodes(n):
    return np.linspace(-1, 1, n)

def chebyshev_nodes(n):
    return np.cos((2 * np.arange(n) + 1) / (2 * n) * np.pi)

# 实验设置
n_values = [2, 4, 8, 16]  # 插值点数量
x = np.linspace(-1, 1, 500)  # 取样点

# 创建子图
fig, axes = plt.subplots(len(n_values), 2, figsize=(12, 18))  # 每行两个子图

for idx, n in enumerate(n_values):
    # 均匀分点
    x_uniform = uniform_nodes(n)
    y_uniform = f(x_uniform)
    L_uniform = lagrange_interpolation(x, x_uniform, y_uniform)

    # 切比雪夫节点
    x_chebyshev = chebyshev_nodes(n)
    y_chebyshev = f(x_chebyshev)
    L_chebyshev = lagrange_interpolation(x, x_chebyshev, y_chebyshev)

    # 绘制均匀分点的插值结果
    ax1 = axes[idx, 0]
    ax1.plot(x, f(x), label="原函数 $f(x)$", color="black")
    ax1.plot(x, L_uniform, label=f"均匀分点 $L_{n}(x)$", linestyle="--", color="blue")
    ax1.scatter(x_uniform, y_uniform, label="均匀分点", color="blue", zorder=5)
    ax1.set_title(f"均匀分点插值结果 (n={n})", pad=10)  # 增加标题与图的距离
    ax1.legend()
    ax1.grid()
    ax1.tick_params(axis='x', labelrotation=45)  # 旋转 x 轴标签

    # 绘制切比雪夫节点的插值结果
    ax2 = axes[idx, 1]
    ax2.plot(x, f(x), label="原函数 $f(x)$", color="black")
    ax2.plot(x, L_chebyshev, label=f"切比雪夫节点 $L_{n}(x)$", linestyle="--", color="red")
    ax2.scatter(x_chebyshev, y_chebyshev, label="切比雪夫节点", color="red", zorder=5)
    ax2.set_title(f"切比雪夫节点插值结果 (n={n})", pad=10)  # 增加标题与图的距离
    ax2.legend()
    ax2.grid()
    ax2.tick_params(axis='x', labelrotation=45)  # 旋转 x 轴标签

# 调整子图布局
plt.tight_layout()
fig.subplots_adjust(hspace=0.4, wspace=0.3)  # 手动调整子图间距
plt.show()