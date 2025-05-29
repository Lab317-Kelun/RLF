import numpy as np
import pyLasaDataset as lasa
from scipy.optimize import minimize
from torch.nn import Parameter
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import math

np.random.seed(0)
torch.manual_seed(0)

class EnergyFunction(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=10, x_set=None, l=10):
        super(EnergyFunction, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l = l
        self.w = Parameter(torch.randn(self.l, 3))
        self.u = Parameter(torch.randn(self.l, self.input_dim) * 0)
        self.w0 = Parameter(torch.randn(1, 3))
        self.weight = Parameter(torch.randn(1, self.l + 1) * 0 + 1)

        self.overline_x = torch.max(torch.sum(x_set ** 2, dim=1) ** 0.5)
        self.scale = self.overline_x / 2

    def forward(self, x):
        weight = torch.exp(self.weight)
        weight = weight/torch.sum(weight)

        a0 = torch.exp(self.w0[0, 0]) + 1e-10
        b0 = self.w0[0, 1]
        c0 = self.w0[0, 1]
        d0 = b0**2 / a0 + torch.exp(self.w0[0, 2])

        a = torch.exp(self.w[:, 0]) + 1e-10
        b = self.w[:, 1]
        c = self.w[:, 1]
        d = b**2 / a + torch.exp(self.w[:, 2])
        t1 = torch.cat((a.unsqueeze(1), b.unsqueeze(1)), dim=1)
        t2 = torch.cat((c.unsqueeze(1), d.unsqueeze(1)), dim=1)
        A = torch.cat((t1.unsqueeze(2), t2.unsqueeze(2)), dim=2)

        x = x.reshape(-1, self.input_dim) / self.scale
        y = (x.unsqueeze(1).unsqueeze(1) @ A.unsqueeze(0) @ (x.unsqueeze(1).unsqueeze(3) - self.u.unsqueeze(0).unsqueeze(3))).squeeze()
        beta = torch.relu(y)/(y + 1e-18)
        v1 = torch.sum(beta.reshape(x.shape[0], -1) * (y**2).reshape(x.shape[0], -1) * weight[0, 1:].reshape(1, -1), dim=1)
        x0 = x[:, 0]
        x1 = x[:, 1]
        v2 = (x0**2)*a0 + x0*x1*(b0 + c0) + (x1**2)*d0
        return v1 + v2 * weight[0,0]

    def forward_numpy(self, x):
        x = torch.tensor(x, dtype=torch.float)
        x = x.reshape(-1, self.input_dim)
        x = self.forward(x).detach().numpy()
        return x

    def jacobian(self, x):
        dt = 0.001
        x = x.reshape(-1, self.input_dim)
        y = self.forward(x)
        dydx = torch.empty(x.shape[0], self.input_dim)
        for i in range(self.input_dim):
            x_ = x.clone()
            x_[:, i] += dt
            y_ = self.forward(x_)
            dydx[:, i] = (y_ - y)/dt
        return dydx

    def jacobian_numpy(self, x):
        dt = 0.001
        x = torch.tensor(x, dtype=torch.float)
        x = x.reshape(-1, self.input_dim)
        y = self.forward(x)
        dydx = torch.empty(x.shape[0], self.input_dim)
        for i in range(self.input_dim):
            x_ = x.clone()
            x_[:, i] += dt
            y_ = self.forward(x_)
            dydx[:, i] = (y_ - y)/dt
        return dydx.detach().numpy()

    def plot_v(self, data_x, x, y, num_levels=10, flat=0):

        X, Y = np.meshgrid(x, y)
        Z = self.forward_numpy(np.column_stack((X.reshape(-1, 1), Y.reshape(-1, 1)))).reshape(np.shape(X))

        if flat==0:
            plt.contourf(X, Y, Z, levels=50, cmap='viridis')

        return X, Y, Z

class LearnClf(nn.Module):
    '''Stable DS learner implemented using pytorch'''
    def __init__(self, manually_design_set, input_dim=2, regularization_param=0.001, hidden_dim=100, ods_hidden_num=200, l=10):
        super(LearnClf, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.manually_design_set = manually_design_set

        data_x, data_y, data_t = manually_design_set
        self.data_x = torch.tensor(np.reshape(data_x, (-1, input_dim)), dtype=torch.float)
        self.data_y = torch.tensor(np.reshape(data_y, (-1, input_dim)), dtype=torch.float)
        self.data_t = torch.tensor(np.reshape(data_t, (-1)), dtype=torch.float)

        self.energy_function = EnergyFunction(self.input_dim, self.hidden_dim, self.data_x, l)

        self.ods_learner = FCNN(self.input_dim, ods_hidden_num, self.input_dim)
        self.regularization_param = regularization_param

        self.k = 0.001
        self.p = 0.1

    def rho(self, x):
        x = x.reshape(-1, self.input_dim)
        x_norm = torch.sum(x**2, dim=1)**0.5
        return self.p * (1 - torch.exp(-x_norm * self.k))


    def forward(self, x):
        dvdx = self.energy_function.jacobian(x)
        ox = self.ods_learner(x)
        u = torch.unsqueeze(-torch.relu(torch.sum(dvdx * ox, dim=1) + self.rho(x)) / torch.sum(dvdx ** 2, dim=1) ,1) * dvdx
        y = ox + u
        return y

    def forward_ext(self, x, ox):
        dvdx = self.energy_function.jacobian(x)
        u = torch.unsqueeze(-torch.relu(torch.sum(dvdx * ox, dim=1) + self.rho(x)) / torch.sum(dvdx ** 2, dim=1) ,1) * dvdx
        y = ox + u
        return y

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float)
        y = self.forward(x)
        return y.detach().numpy()

    def predict_ext(self, x, ox):
        x = torch.tensor(x, dtype=torch.float)
        ox = torch.tensor(ox, dtype=torch.float).reshape(x.shape)
        y = self.forward_ext(x, ox)
        return y.detach().numpy()

    def train_ods(self, savepath, epochs=1000, lr_=0.01):
        self.train()

        loss_function = torch.nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr_)

        input_tensor = self.data_x
        output_tensor = self.data_y
        loss_min = 1e18

        for epoch in range(epochs):
            optimizer.zero_grad()
            predicted_output = self.ods_learner(input_tensor)

            loss = loss_function(predicted_output, output_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

            if loss.item() < loss_min:
                loss_min = loss.item()
                torch.save(self.state_dict(), savepath)

        self.load_state_dict(torch.load(savepath))

    def train_energy(self, savepath, epochs=1000, lr_=0.01):
        self.train()

        loss_function = torch.nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr_)

        input_tensor = self.data_x
        output_tensor = self.data_y
        loss_min = 1e18

        for epoch in range(epochs):
            optimizer.zero_grad()
            predicted_output = self(input_tensor)

            dvdx = self.energy_function.jacobian(input_tensor)
            dot_x = self.data_y

            eps = 1e-8
            a_norm = dot_x / (torch.norm(dot_x, dim=1, keepdim=True) + eps)
            b_norm = dvdx / (torch.norm(dvdx, dim=1, keepdim=True) + eps)
            dot_products = torch.sum(a_norm * b_norm, dim=1)

            L1 = torch.sum(torch.tanh(dot_products*10)) / dot_products.shape[0] + 1
            loss = L1
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {L1.item()}')

            if L1.item() < loss_min:
                loss_min = loss.item()
                torch.save(self.state_dict(), savepath)

        self.load_state_dict(torch.load(savepath))

    def train_all(self, savepath, epochs=2000, lr_=0.005):
        self.train()

        loss_function = torch.nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr_)

        input_tensor = self.data_x
        output_tensor = self.data_y
        loss_min = 1e18

        for epoch in range(epochs):
            optimizer.zero_grad()

            predicted_output = self(input_tensor)
            loss = loss_function(predicted_output, output_tensor)

            l2_reg = torch.tensor(0.)
            for param in self.parameters():
                l2_reg += torch.norm(param, 2)
            loss += self.regularization_param * l2_reg

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

            if loss.item() < loss_min:
                loss_min = loss.item()
                torch.save(self.state_dict(), savepath)

        self.load_state_dict(torch.load(savepath))

class RFFN(nn.Module):
    '''Random Fourier features network.'''
    def __init__(self, in_dim, nfeat, out_dim, sigma=10):
        super(RFFN, self).__init__()
        self.sigma = np.ones(in_dim) * sigma
        self.coeff = np.random.normal(0.0, 1.0, (nfeat, in_dim))
        self.coeff = self.coeff / self.sigma.reshape(1, len(self.sigma))
        self.offset = 2.0 * np.pi * np.random.rand(1, nfeat)

        self.network = nn.Sequential(
            LinearClamped(in_dim, nfeat, self.coeff, self.offset),
            Cos(),
            nn.Linear(nfeat, out_dim, bias=False)
        )

    def forward(self, x):
        return self.network(x)


class FCNN(nn.Module):
    '''2-layer fully connected neural network'''
    def __init__(self, in_dim, hidden_dim, out_dim, act='tanh'):
        super(FCNN, self).__init__()
        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'leaky_relu': nn.LeakyReLU,
                        'elu': nn.ELU, 'prelu': nn.PReLU, 'softplus': nn.Softplus}
        act_func = activations[act]
        self.network = nn.Sequential(
			nn.Linear(in_dim, hidden_dim), act_func(),
			nn.Linear(hidden_dim, hidden_dim), act_func(),
			nn.Linear(hidden_dim, out_dim)
		)

    def forward(self, x):
        return self.network(x)

class FCNN_Single(nn.Module):
    '''1-layer fully connected neural network'''
    def __init__(self, in_dim, hidden_dim, out_dim, act='tanh'):
        super(FCNN_Single, self).__init__()
        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'leaky_relu': nn.LeakyReLU,
                        'elu': nn.ELU, 'prelu': nn.PReLU, 'softplus': nn.Softplus}
        act_func = activations[act]
        self.network = nn.Sequential(
			nn.Linear(in_dim, hidden_dim), act_func(),
			nn.Linear(hidden_dim, out_dim)
		)

    def forward(self, x):
        return self.network(x)

class LinearClamped(nn.Module):
	'''Linear layer with user-specified parameters (not to be learrned!)'''
	__constants__ = ['bias', 'in_features', 'out_features']
	def __init__(self, in_features, out_features, weights, bias_values, bias=True):
		super(LinearClamped, self).__init__()
		self.in_features = in_features
		self.out_features = out_features

		self.register_buffer('weight', torch.Tensor(weights))
		if bias:
			self.register_buffer('bias', torch.Tensor(bias_values))

	def forward(self, input):
		if input.dim() == 1:
			return F.linear(input.view(1, -1), self.weight, self.bias)
		return F.linear(input, self.weight, self.bias)

	def extra_repr(self):
		return 'in_features={}, out_features={}, bias={}'.format(
			self.in_features, self.out_features, self.bias is not None
		)

class Cos(nn.Module):
	'''Applies the cosine element-wise function'''
	def forward(self, inputs):
		return torch.cos(inputs)