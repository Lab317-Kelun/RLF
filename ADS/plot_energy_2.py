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
y1 = x
y2 = 2 * x
y3 = 3 * x

z1 = x**2 + y1**2
z2 = x**2 + y2**2
z3 = x**2 + y3**2

x_1 = np.cos(z1*0.001)*x + np.sin(0.001*z1)*y1
x_2 = np.cos(z2*0.001)*x + np.sin(0.001*z2)*y2
x_3 = np.cos(z3*0.001)*x + np.sin(0.001*z3)*y3

y_1 = -np.sin(z1*0.001)*x + np.cos(0.001*z1)*y1
y_2 = -np.sin(z2*0.001)*x + np.cos(0.001*z2)*y2
y_3 = -np.sin(z3*0.001)*x + np.cos(0.001*z3)*y3

plt.plot(x, y1,  color="blue", linewidth=2, linestyle='--')
plt.plot(x, y2,  color="green", linewidth=2, linestyle='--')
plt.plot(x, y3,  color="red", linewidth=2, linestyle='--')


plt.plot(x_1, y_1, color="blue", linewidth=2, linestyle='-')
plt.plot(x_2, y_2,  color="green", linewidth=2, linestyle='-')
plt.plot(x_3, y_3, color="red", linewidth=2, linestyle='-')

plt.xlim(-25, 25)  # 设置 x 轴的范围从 0 到 12
plt.ylim(-25, 25)  # 设置 y 轴的范围从 -1.5 到 1.5

plt.grid(True, which='both', axis='both', linestyle='-', color='gray', linewidth=0.5)
plt.savefig('P3.png', dpi=300)
