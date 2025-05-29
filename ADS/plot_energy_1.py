import numpy as np
import matplotlib.pyplot as plt
import pyLasaDataset as lasa
from scipy.linalg import norm, pinv
from Algorithms.Learn_SDS import LearnSds
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
from torch.autograd import grad
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import savemat
import time

Type = 'Snake'
data = getattr(lasa.DataSet, Type)
demos = data.demos

def construct_demonstration_set(demos, start=1, end=-1, gap=5, used_tras=[0, 1, 2, 3, 4, 5, 6]):
    n_tra = len(used_tras)
    x_set = []
    dot_x_set = []
    t_set = []
    for i in range(n_tra):
        x_set.append(demos[used_tras[i]].pos[:, start:end:gap].T)
        dot_x_set.append(demos[used_tras[i]].vel[:, start:end:gap].T)
        t_set.append(demos[used_tras[i]].t[0, start:end:gap])

    x_set = np.array(x_set)
    t_set = np.array(t_set)
    dot_x_set = np.array(dot_x_set)
    return x_set, dot_x_set, t_set

data_x, data_y, data_t = construct_demonstration_set(demos, start=20, end=-1, gap=5)
from matplotlib.colors import Normalize
norm = Normalize(vmin=-20, vmax=135)
plt.figure(figsize=(6, 6))
x_ = []
y_ = []
for j in range(np.shape(data_x)[1]):
    x = torch.tensor([data_y[1, j, 0], data_y[1, j, 1]], dtype=torch.float)
    dvdx = torch.tensor([data_x[1, j, 0], data_x[1, j, 1]], dtype=torch.float)
    if torch.sum(dvdx * x) < 0:
        None
    else:
        x_.append(data_x[1, j, 0])
        y_.append(data_x[1, j, 1])
d = 50
min_x = - d
max_x = + d
min_y = - d
max_y = + d
x = np.arange(min_x, max_x, 0.05)
y = np.arange(min_y, max_y, 0.05)
X, Y = np.meshgrid(x, y)
Z = (X)**2 + (Y)**2
plt.xlim(-5, 5)  # 设置 x 轴的范围从 0 到 12
plt.ylim(-5, 5)  # 设置 y 轴的范围从 -1.5 到 1.5
contour = (plt).contour(X, Y, Z, levels=[2], linewidths=1, colors='red', zorder=3, norm=norm)
# plt.clabel(contour, inline=True,  fontsize=8, fmt="%.3f")  # 格式化数值为两位小数
plt.plot(data_x[1, :, 0], data_x[1, :, 1], color='red', linestyle='--', linewidth=4, zorder = 8, alpha=0.8, label='Demonstration data')
plt.scatter(0, 0, color='blue', linewidth=2, zorder = 11, label='Target point')
plt.legend(loc='upper left')

Z = (np.tanh(0.1*((X**2 + Y**2)**0.5)) * X)**2 + (np.tanh(0.1*((X**2 + Y**2)**0.5)) * Y)**2
contour = plt.contour(X, Y, Z, levels=[2], linewidths=1, colors='green', zorder=3, norm=norm, label='green')

the = np.arctan2(Y, X) * 10
a = (np.sin(the)*0.1 + 1)*0.1
Z = (np.tanh(a*((X**2 + Y**2)**0.5)) * X)**2 + (np.tanh(a*((X**2 + Y**2)**0.5)) * Y)**2
contour = plt.contour(X, Y, Z, levels=[2], linewidths=1, colors='blue', zorder=3, norm=norm)

plt.grid(True, which='both', axis='both', linestyle='-', color='gray', linewidth=0.5)
plt.savefig('P2.png', dpi=300)
