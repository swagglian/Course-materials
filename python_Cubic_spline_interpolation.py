import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
def f(x):
    n = len(x)  # 确定传入参数个数
    if n==2:
        return (x[1][1] - x[0][1]) / (x[1][0] - x[0][0])
    x0 = deepcopy(x)
    x1 = deepcopy(x)
    xk1 = x0.pop(n-2)
    xk = x1.pop()

    return (f(x0) - f(x1)) / (xk[0] - xk1[0])
def calculat_parameter1(x, f0, fn):
    n = len(x)
    h = []
    for i in range(1, n):
        h.append(x[i][0] - x[i-1][0])
    u = [1]
    for i in range(1, n-1)[::-1]:
        u.insert(0, h[i-1] / (h[i-1] + h[i]))
    lamda = [1]
    for i in range(1, n-1):
        lamda.append(h[i] / (h[i-1] + h[i]))
    d = []
    for i in range(0, n):
        if i == 0:
            d.append(6 * (f([x[i], x[i+1]]) - f0) / h[0])
        elif i == n-1:
            d.append(6 * (fn - f([x[i-1], x[i]])) / h[i-1])
        else:
            d.append(6*f([x[i-1], x[i], x[i+1]]))

    return h, u, lamda, d
def calculat_parameter2(x, M0, Mn):
    n = len(x)
    h = []
    for i in range(1, n):
        h.append(x[i][0] - x[i-1][0])
    u = [0]
    for i in range(1, n-1)[::-1]:
        u.insert(0, h[i-1] / (h[i-1] + h[i]))
    lamda = [0]
    for i in range(1, n-1):
        lamda.append(h[i] / (h[i-1] + h[i]))
    d = []
    for i in range(0, n):
        if i == 0:
            d.append(2 * M0)
        elif i == n-1:
            d.append(2 * Mn)
        else:
            d.append(6*f([x[i-1], x[i], x[i+1]]))

    return h, u, lamda, d

def calculat_M(lamda, u, d):
    n = len(lamda)
    A = 2 * np.eye(n+1, dtype="float")

    for i in range(n):  # 构造矩阵A
        A[i][i+1] = lamda[i]
        A[i+1][i] = u[i]

    b = np.array([d]).transpose()
    M = np.dot(np.linalg.inv(A), b)
    return M

def s(x0, x1, M0, M1, x):
    h = x1[0] - x0[0]
    return M0 * pow((x1[0] - x), 3) / (6 * h) + M1 * pow((x - x0[0]), 3) / (6 * h) + (x0[1] - M0 * h * h / 6) * (
    x1[0] - x) / h + (x1[1] - M1 * h * h / 6) * (x - x0[0]) / h

def plot_s(x, M, color="#ff0000", label="方式"):
    n = len(x)

    plt.figure(1)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('三次样条插值')
    for i in range(1, n):
        x0, x1 = x[i-1], x[i]
        plt.plot([x0[0], x1[0]], [x0[1], x1[1]], 'xc')
        M0, M1 = M[i-1], M[i]
        x_vector = np.linspace(x0[0], x1[0], 200)
        y_vector = s(x0, x1, M0, M1, x_vector).transpose()
        l, = plt.plot(x_vector, y_vector, color=color, lw=1)
    l.set_label(label) 


x = [(0, 1), (1, 0), (2, 0), (3, 1), (4, 2), (5, 2), (6, 1)]  # x点集

h1, u1, lamda1, d1 = calculat_parameter1(x, -0.6, -1.8)  # 第一类边界
M1 = calculat_M(lamda1, u1, d1)
plot_s(x, M1, color="#a012c5", label="第一类边界")
h2, u2, lamda2, d2 = calculat_parameter2(x, 1, -1)  # 第二类边界
M2 = calculat_M(lamda2, u2, d2)
plot_s(x, M2, color="#05a0ff", label="第二类边界")
h3, u3, lamda3, d3 = calculat_parameter2(x, 0, 0)  