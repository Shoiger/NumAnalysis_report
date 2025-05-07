import numpy as np
import time
from tabulate import tabulate

# 定义微分方程
def f(x, y):
    return -y + np.cos(2 * x) - 2 * np.sin(2 * x) + 2 * x * np.exp(-x)

# 精确解
def exact_solution(x):
    return x * np.exp(-x) + np.cos(2 * x)

# Runge-Kutta 四阶方法
def runge_kutta_4(f, x0, y0, h, n):
    x = np.linspace(x0, x0 + n * h, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0

    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h / 2, y[i] + h / 2 * k1)
        k3 = f(x[i] + h / 2, y[i] + h / 2 * k2)
        k4 = f(x[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return x, y

# 修正的 Adams 预测-校正法
def adams_bashforth_moulton(f, x0, y0, h, n):
    x = np.linspace(x0, x0 + n * h, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0

    # 使用 RK4 初始化前四步
    for i in range(3):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h / 2, y[i] + h / 2 * k1)
        k3 = f(x[i] + h / 2, y[i] + h / 2 * k2)
        k4 = f(x[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # 预测-校正法
    for i in range(3, n):
        # 预测
        yp = y[i] + h / 24 * (55 * f(x[i], y[i]) - 59 * f(x[i - 1], y[i - 1]) +
                              37 * f(x[i - 2], y[i - 2]) - 9 * f(x[i - 3], y[i - 3]))
        # 修正
        yc = y[i] + h / 24 * (9 * f(x[i + 1], yp) + 19 * f(x[i], y[i]) -
                              5 * f(x[i - 1], y[i - 1]) + f(x[i - 2], y[i - 2]))
        y[i + 1] = yc

    return x, y

# 参数设置
x0, y0 = 0, 1
steps = [0.1, 0.001]  # 两个步长

for h in steps:
    n = int(2 / h)

    # 计算时间
    start_time = time.time()
    x_rk, y_rk = runge_kutta_4(f, x0, y0, h, n)
    rk_time = time.time() - start_time

    start_time = time.time()
    x_adams, y_adams = adams_bashforth_moulton(f, x0, y0, h, n)
    adams_time = time.time() - start_time

    # 精确解
    y_exact = exact_solution(x_rk)

    # 误差计算
    rk_error = np.abs(y_rk - y_exact)
    adams_error = np.abs(y_adams - y_exact)

    # 表格打印（显示起始、中间和结尾部分节点）
    display_indices = np.concatenate(([0], np.linspace(1, len(x_rk) // 2, 3, dtype=int), [len(x_rk) - 1]))
    table_headers = ["x", "精确解", "Runge-Kutta 结果", "Runge-Kutta 误差", "Adams 结果", "Adams 误差"]
    table_data = []

    for i in display_indices:
        table_data.append([x_rk[i], y_exact[i], y_rk[i], rk_error[i], y_adams[i], adams_error[i]])

    print(f"\n计算结果表格（步长 h={h}）：")
    print(tabulate(table_data, headers=table_headers, floatfmt=".6f", tablefmt="grid"))

    # 打印计算时间
    print(f"\nRunge-Kutta 方法计算时间: {rk_time:.6f} 秒")
    print(f"Adams 方法计算时间: {adams_time:.6f} 秒")