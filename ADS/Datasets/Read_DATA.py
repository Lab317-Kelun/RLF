import numpy as np
import matplotlib.pyplot as plt
import pyLasaDataset as lasa
from scipy.linalg import norm, pinv
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
import torch
import torch.nn as nn
from torch.autograd import grad
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import savemat
import time
import scipy.io
import numpy as np
import time
import pyLasaDataset as lasa
np.random.seed(5)

fig1 = plt.figure()
data_x = np.zeros((5, 50, 2))
data_y = data_x.copy()
data_t = np.zeros((5, 50))
dt = 0.01

result = np.array([[-1.0, 2.0],[-1.0, 3.0],[-2.0, 3.0], [-2.0, 2.0], [-2.0, 1.0], [-2.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
original_indices = np.linspace(0, 1, result.shape[0])  # 原始数据的索引
target_indices = np.linspace(0, 1, 51)  # 目标插值的索引
interpolator = interp1d(original_indices, result, kind='quadratic', axis=0)
result = interpolator(target_indices)
data_x[0, :, :] = result[0:-1, :]
data_y[0, :, :] = np.diff(result, axis=0) / dt
data_t[0, :] = np.arange(0, dt * 50, dt)

d = 0.02
result = np.array([[-1.0 + d, 2.0],[-1.0 + d, 3.0 + d],[-2.0 - d, 3.1], [-2.0 - d, 2.0], [-2.0 - d, 1.0], [-2.0 - d, 0.0], [-1.0, 0.0], [0.0, 0.0]])
original_indices = np.linspace(0, 1, result.shape[0])
target_indices = np.linspace(0, 1, 51)
interpolator = interp1d(original_indices, result, kind='quadratic', axis=0)
result = interpolator(target_indices)
data_x[1, :, :] = result[0:-1, :]
data_y[1, :, :] = np.diff(result, axis=0) / dt
data_t[1, :] = np.arange(0, dt * 50, dt)

d = 0.04
result = np.array([[-1.0 + d, 2.0],[-1.0 + d, 3.0 + d],[-2.0 - d, 3.1], [-2.0 - d, 2.0], [-2.0 - d, 1.0], [-2.0 - d, 0.0], [-1.0, 0.0], [0.0, 0.0]])
original_indices = np.linspace(0, 1, result.shape[0])
target_indices = np.linspace(0, 1, 51)
interpolator = interp1d(original_indices, result, kind='quadratic', axis=0)
result = interpolator(target_indices)
data_x[2, :, :] = result[0:-1, :]
data_y[2, :, :] = np.diff(result, axis=0) / dt
data_t[2, :] = np.arange(0, dt * 50, dt)

d = 0.06
result = np.array([[-1.0 + d, 2.0],[-1.0 + d, 3.0 + d],[-2.0 - d, 3.1], [-2.0 - d, 2.0], [-2.0 - d, 1.0], [-2.0 - d, 0.0], [-1.0, 0.0], [0.0, 0.0]])
original_indices = np.linspace(0, 1, result.shape[0])
target_indices = np.linspace(0, 1, 51)
interpolator = interp1d(original_indices, result, kind='quadratic', axis=0)
result = interpolator(target_indices)
data_x[3, :, :] = result[0:-1, :]
data_y[3, :, :] = np.diff(result, axis=0) / dt
data_t[3, :] = np.arange(0, dt * 50, dt)

d = 0.08
result = np.array([[-1.0 + d, 2.0],[-1.0 + d, 3.0 + d],[-2.0 - d, 3.1], [-2.0 - d, 2.0], [-2.0 - d, 1.0], [-2.0 - d, 0.0], [-1.0, 0.0], [0.0, 0.0]])
original_indices = np.linspace(0, 1, result.shape[0])
target_indices = np.linspace(0, 1, 51)
interpolator = interp1d(original_indices, result, kind='quadratic', axis=0)
result = interpolator(target_indices)
data_x[4, :, :] = result[0:-1, :]
data_y[4, :, :] = np.diff(result, axis=0) / dt
data_t[4, :] = np.arange(0, dt * 50, dt)

data_x = (data_x) * 10
for i in range(np.shape(data_x)[0]):
    for j in range(np.shape(data_x)[1]):
        plt.scatter(data_x[i, j, 0], data_x[i, j, 1], color='red', zorder=1, s=5)
plt.show()

np.savez('data_set_1.npz', data_x=data_x, data_y=data_y, data_t=data_t)

