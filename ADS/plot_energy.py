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
d = 5
min_x = np.min(data_x[:, :, 0]) - d
max_x = np.max(data_x[:, :, 0]) + d
min_y = np.min(data_x[:, :, 1]) - d
max_y = np.max(data_x[:, :, 1]) + d
x = np.arange(min_x, max_x, 1)
y = np.arange(min_y, max_y, 1)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2
from matplotlib.colors import Normalize
# contourf = plt.contourf(X, Y, Z, levels=20, linewidths=1, cmap='viridis', zorder=2)  # 绘制等高线
contour = plt.contour(X, Y, Z, levels=20, linewidths=1, colors='lightblue', zorder=3)
# plt.clabel(contourf, inline=True, fontsize=8, fmt="%.3f")  # 格式化数值为两位小数

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
plt.plot(x_, y_, color='red', zorder = 10, linewidth=1.5, label='Violating data')
plt.plot(data_x[1, :, 0], data_x[1, :, 1], color='green', linewidth=1.5, zorder = 9, label='Consistent data')
plt.plot(data_x[1, :, 0], data_x[1, :, 1], color='yellow',  linestyle='--', linewidth=4, zorder = 8, alpha=0.8, label='Demonstration data')
plt.scatter(0, 0, color='blue', linewidth=2, zorder = 11, label='Target point')
plt.legend(loc='upper left')
plt.savefig('P1.png', dpi=1000)

