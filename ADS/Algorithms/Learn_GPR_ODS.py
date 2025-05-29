'''
Learning the original ADS using Gaussian Process.
Details can be found in the paper:
"Learning a flexible neural energy function with a unique minimum
for stable and accurate demonstration learning"
'''

import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
import autograd.scipy.stats.multivariate_normal as mvn
from autograd.numpy.linalg import solve

np.random.seed(5)


class SingleGpr:
    def __init__(self, X, y, observation_noise=None, gamma=0.5):
        '''
        Initializing the single GPR
        :param X: Input set, (input_num * input_dim)
        :param y: Output set for one dim, (n_size,)
        :param observation_noise: the standard deviation for observation_noise
        :param gamma: a scalar which will be used if observation_noise is None
        '''
        self.input_dim = np.shape(X)[1]
        # add the equilibrium-point (0, 0) to the set
        self.X = np.vstack((X, np.zeros(self.input_dim).reshape(1, -1)))
        self.y = np.hstack((y, np.array([0.0])))
        self.input_num = np.shape(self.X)[0]
        if observation_noise is None:
            self.observation_noise = gamma * np.sqrt(np.average(self.y**2))
        else:
            self.observation_noise = observation_noise
        self.param = self.init_random_param()
        # Used to tune the degree of the equilibrium-point confidence
        self.determined_point_degree = 0.0

    def init_random_param(self):
        '''
        Initializing the hyper-parameters
        '''
        sqrt_kernel_length_scale = np.sqrt(np.diag(np.cov(self.X.T)))
        kernel_noise = np.sqrt(np.average(self.y**2))
        param = np.hstack((kernel_noise, sqrt_kernel_length_scale))
        return param

    def set_param(self, param):
        '''
        Manually set the hyper-parameters
        '''
        self.param = param.copy()
        '''
        pre-computations for prediction
        '''
        self.cov_y_y = self.rbf(self.X, self.X, self.param)
        temp = self.observation_noise ** 2 * np.eye(self.input_num)
        # observation noises for determined set should be zero
        temp[-1, -1] = self.determined_point_degree
        self.cov_y_y = self.cov_y_y + temp
        self.beta = solve(self.cov_y_y, self.y)  # The constant vector of the mean prediction function
        self.inv_cov_y_y = solve(self.cov_y_y, np.eye(self.input_num))

    def build_objective(self, param):
        '''
        The obj of Single GPR
        '''
        cov_y_y = self.rbf(self.X, self.X, param)
        temp = self.observation_noise**2 * np.eye(self.input_num)
        # observation noises for determined set should be zero
        temp[-1, -1] = self.determined_point_degree
        cov_y_y = cov_y_y + temp
        out = - mvn.logpdf(self.y, np.zeros(self.input_num), cov_y_y)
        return out

    def train(self):
        '''
        Training Single GPR
        '''
        result = minimize(value_and_grad(self.build_objective), self.param, jac=True, method='L-BFGS-B', tol=1e-8,
                          options={'maxiter': 50, 'disp': False})
        self.param = result.x
        # pre-computation for prediction
        self.cov_y_y = self.rbf(self.X, self.X, self.param)
        temp = self.observation_noise ** 2 * np.eye(self.input_num)
        temp[-1, -1] = self.determined_point_degree
        self.cov_y_y = self.cov_y_y + temp
        self.beta = solve(self.cov_y_y, self.y)
        self.inv_cov_y_y = solve(self.cov_y_y, np.eye(self.input_num))

    def rbf(self, x, x_, param):
        '''
        Construct the kernel matrix (or scalar, vector)
        '''
        kn = param[0]  # abbreviation for kernel_noise
        sqrt_kls = param[1:]  # abbreviation for sqrt_kernel_length_scale
        '''
        Using the broadcast technique to accelerate computation
        '''
        diffs = np.expand_dims(x / sqrt_kls, 1) - np.expand_dims(x_ / sqrt_kls, 0)
        return kn ** 2 * np.exp(-0.5 * np.sum(diffs ** 2, axis=2))

    def predict_determined_input(self, inputs):
        '''
        Prediction of Single GPR
        '''
        cov_y_f = self.rbf(self.X, inputs, self.param)
        means = np.dot(cov_y_f.T, self.beta)  # (m,)
        return means


class MultiGpr:
    def __init__(self, X, Y, observation_noise=None, gamma=0.5):
        '''
        MultiGPR is a stack of multiple single GPRs;
        :param X: Input set, (input_num * input_dim)
        :param Y: Output set, (input_num * output_dim)
        :param observation_noise: the standard deviation for observation_noise
        '''
        self.X = X
        self.Y = Y
        self.input_dim = np.shape(X)[1]
        self.input_num = np.shape(X)[0]
        self.output_dim = np.shape(Y)[1]
        self.observation_noise = observation_noise
        self.gamma = gamma
        self.models = self.create_models()

    def set_param(self, param):
        '''
        Manually set the parameter
        '''
        for i in range(self.output_dim):
            self.models[i].set_param(param[i])

    def create_models(self):
        '''
        Creating a stack of single GPR
        '''
        models = []
        for i in range(self.output_dim):
            if self.observation_noise is not None:
                models.append(SingleGpr(self.X, self.Y[:, i], observation_noise=self.observation_noise[i], gamma=self.gamma))
            else:
                models.append(SingleGpr(self.X, self.Y[:, i], observation_noise=None, gamma=self.gamma))
        return models

    def train(self, save_path=None):
        '''
        Training multi-PGR
        '''
        for i in range(self.output_dim):
            print('training model ', i, '...')
            self.models[i].train()
            if i == 0:
                param = self.models[i].param.copy()
            else:
                param = np.vstack((param, self.models[i].param.copy()))
        if save_path is not None:
            np.savetxt(save_path, param)

    def predict_determined_input(self, inputs):
        '''
        Prediction of Multi-GPR
        '''
        for i in range(self.output_dim):
            if i == 0:
                means = self.models[0].predict_determined_input(inputs).copy()
            else:
                means = np.vstack((means, self.models[i].predict_determined_input(inputs).copy()))
        means = means.T
        return means  # (m, output_dimension)


class LearnOds:
    def __init__(self, manually_design_set, observation_noise=None, gamma=0.5):
        '''
        Initializing the original ADS
        :param manually_design_set: (x_set, dot_x_set, t_set)
        :param observation_noise: the standard deviation for observation_noise
        :param gamma: a scalar which will be used if observation_noise is None
        '''
        self.x_set, self.dot_x_set, _ = manually_design_set
        self.d_x = np.shape(self.x_set)[2]
        if np.isscalar(observation_noise):
            self.MultiGpr = MultiGpr(self.x_set.reshape(-1, self.d_x), self.dot_x_set.reshape(-1, self.d_x), np.ones(self.d_x) * observation_noise, gamma=gamma)
        else:
            self.MultiGpr = MultiGpr(self.x_set.reshape(-1, self.d_x), self.dot_x_set.reshape(-1, self.d_x), observation_noise, gamma=gamma)

    def set_param(self, param):
        '''
        Set parameters of the original ADS
        '''
        self.MultiGpr.set_param(param)

    def train(self, save_path=None):
        '''
        Training the original ADS
        '''
        self.MultiGpr.train(save_path=save_path)

    def predict(self, inputs):
        '''
        Prediction of the orginal ADS
        '''
        outputs = self.MultiGpr.predict_determined_input(inputs)
        return outputs

