import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm, pinv
from Algorithms.Learn_GPR_ODS import LearnOds
from scipy.io import savemat
import openpyxl
import os
import torch
import similaritymeasures as sm

np.random.seed(0)
torch.manual_seed(0)

def DTWDistance(s1, s2):
    # s1/s2, (n, 2) np.array
    DTW = {}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = np.sum((s1[i] - s2[j]) ** 2)
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return np.sqrt(DTW[len(s1) - 1, len(s2) - 1])

class LearnSds():
    def __init__(self, manually_design_set, clf_learner, Type):
        super(LearnSds, self).__init__()
        self.manually_design_set = manually_design_set
        self.clf_learner = clf_learner
        self.ods_learner = LearnOds(manually_design_set=manually_design_set, observation_noise=None, gamma=0.5)
        # self.ods_learner.train('OadsParameters/Oads_parameter_for_' + Type + '.txt')
        oads_parameters = np.loadtxt('OadsParameters/Oads_parameter_for_' + Type + '.txt')
        self.ods_learner.set_param(oads_parameters)

    def simutation(self, save_path, Type, cnt_, tim, Method, gpr):
        dt = 0.01
        eta1 = 1
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        data_x, data_y, data_t = self.manually_design_set
        data_x, data_y, data_t = self.manually_design_set
        data_x_pre = np.zeros((np.shape(data_x)[0], 50, 2))

        d = 5
        min_x = np.min(data_x[:, :, 0]) - d
        max_x = np.max(data_x[:, :, 0]) + d
        min_y = np.min(data_x[:, :, 1]) - d
        max_y = np.max(data_x[:, :, 1]) + d
        px, py, pz = self.clf_learner.energy_function.plot_v(data_x, np.arange(min_x, max_x, 0.1), np.arange(min_y, max_y, 0.1))
        print(pz)
        for i in range(np.shape(data_x)[0]):
            t = 0
            x = []
            y = []
            Pos = np.array([data_x[i, 0, 0], data_x[i, 0, 1]])
            cnt = 0
            while True:
                t = t + dt
                if abs(Pos).sum() < eta1:
                    print(i)
                    print(Pos)
                    break
                if cnt > 2000:
                    print(i)
                    print(Pos)
                    break
                cnt = cnt + 1
                if gpr == False:
                    Pos_Dot = self.clf_learner.predict(Pos).flatten()
                elif gpr == True:
                    Pos_Dot = self.clf_learner.predict_ext(Pos, self.ods_learner.predict(Pos)).flatten()
                x.append(Pos[0])
                y.append(Pos[1])
                Pos = Pos + Pos_Dot * dt

            sampled_array = np.linspace(start=0, stop=len(x) - 1, num=50, dtype=int)
            x = np.array(x)
            y = np.array(y)
            x = x[sampled_array]
            y = y[sampled_array]
            data_x_pre[i, 0:50, 0] = x
            data_x_pre[i, 0:50, 1] = y

        indices = torch.linspace(0, data_x.shape[1] - 1, 50).long()
        data_x = data_x[:, indices, :]
        data_y = data_y[:, indices, :]

        X, Y = np.meshgrid(np.arange(min_x, max_x, 1), np.arange(min_y, max_y, 1))
        if gpr == False:
            J = self.clf_learner.predict(np.column_stack((X.reshape(-1, 1), Y.reshape(-1, 1))))
        elif gpr == True:
            J = self.clf_learner.predict_ext(np.column_stack((X.reshape(-1, 1), Y.reshape(-1, 1))), self.ods_learner.predict(np.column_stack((X.reshape(-1, 1), Y.reshape(-1, 1)))))

        U = J[:, 0].reshape(np.shape(X))
        V = J[:, 1].reshape(np.shape(X))

        for i in range(np.shape(data_x)[0]):
            for j in range(np.shape(data_x)[1]):
                dvdx = self.clf_learner.energy_function.jacobian(torch.tensor([data_x[i, j, 0], data_x[i, j, 1]], dtype=torch.float))
                x = torch.tensor([data_y[i, j, 0], data_y[i, j, 1]], dtype=torch.float)
                if torch.sum(dvdx * x)<0:
                    ax.scatter(data_x[i, j, 0], data_x[i, j, 1], color='blue', zorder=1, s=5)
                else:
                    ax.scatter(data_x[i, j, 0], data_x[i, j, 1], color='red', zorder=1, s=5)

        ax.scatter(0, 0, c='black', alpha=1.0, s=50, marker='X')
        ax.scatter(data_x[:, 0, 0], data_x[:, 0, 1], c='black', alpha=1.0, s=10, marker='o')

        ax.set_xticks([])
        ax.set_yticks([])
        ax = plt.gca()
        ax.set_xlim(min_x + 2, max_x - 2)
        ax.set_ylim(min_y + 2, max_y - 2)
        plt.scatter(0, 0, color='black', marker='x', label='Point at (0, 0)', s=150, linewidth=5, zorder=10)
        plt.savefig(save_path + '1.png')
        plt.savefig(save_path + '1.eps')

        fig2 = plt.figure()
        ax = fig2.add_subplot(111)
        self.clf_learner.energy_function.plot_v(data_x, np.arange(min_x, max_x, 0.1), np.arange(min_y, max_y, 0.1))

        for i in range(data_x.shape[0]):
            plt.plot(data_x[i, :, 0], data_x[i, :, 1], color='white', linewidth=3, zorder=2)
        for i in range(data_x.shape[0]):
            plt.plot(data_x_pre[i, :, 0], data_x_pre[i, :, 1], color='red', linewidth=3, zorder=2)

        plt.streamplot(X, Y, U, V, density=1.0, linewidth=0.6, maxlength=1.0, minlength=0.1,
                    arrowstyle='simple', arrowsize=1, zorder=1)

        ax.set_xticks([])
        ax.set_yticks([])
        ax = plt.gca()
        ax.set_xlim(min_x + 2, max_x - 2)
        ax.set_ylim(min_y + 2, max_y - 2)
        plt.scatter(0, 0, color='black', marker='x', label='Point at (0, 0)', s=150, linewidth=5, zorder=10)
        plt.savefig(save_path + '2.png')
        plt.savefig(save_path + '2.eps')

        save_path = 'Result_mat_' + Method + '/' + Type + '.mat'
        savemat(save_path, {'odata': data_x ,'data': data_x_pre, 'px': px, 'py': py, 'pz': pz, 'lx': X, 'ly': Y, 'lu': U, 'lv': V})

        parent_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        data_root = parent_directory + '\data_' + Method + '.xlsx'
        print(data_root)
        workbook = openpyxl.load_workbook(data_root)
        sheet = workbook.active

        sheet['H' + str(cnt_)] = str(round(tim, 2))
        for i in range(data_x.shape[0]):
            sheet['A' + str(cnt_)] = Type

            if gpr == False:
                ox = self.clf_learner.predict(data_x[i, :, :])
            elif gpr == True:
                ox = self.clf_learner.predict_ext(data_x[i, :, :], self.ods_learner.predict(data_x[i, :, :]))

            Vrmse = (ox - data_y[i, :, :])**2
            Vrmse = np.sum(Vrmse, axis=1)
            Vrmse = np.sum(Vrmse)/Vrmse.shape[0]
            Vrmse = Vrmse**0.5
            sheet['C' + str(cnt_)] = str(round(Vrmse, 2))

            rmse = (data_x[i, :, :] - data_x_pre[i, :, :])**2
            rmse = np.sum(rmse, axis=1)
            rmse = np.sum(rmse)/rmse.shape[0]
            rmse = rmse**0.5
            sheet['D' + str(cnt_)] = str(round(rmse, 2))

            dvdx = self.clf_learner.energy_function.jacobian(torch.tensor(data_x[i, :, :], dtype=torch.float))
            dx = torch.tensor(data_y[i, :, :], dtype=torch.float)
            V = torch.sum(dx*dvdx, dim=1)/((torch.sum(dx**2, dim=1)*torch.sum(dvdx**2, dim=1))**0.5 + 0.001)
            V = torch.sum(torch.tanh(V*10))/V.shape[0]
            sheet['E' + str(cnt_)] = str(np.round(V.detach().numpy(), 2))

            dtw, d = sm.dtw(data_x[i, :, :], data_x_pre[i, :, :])
            sheet['F' + str(cnt_)] = str(round(dtw, 2))

            df = sm.frechet_dist(data_x[i, :, :], data_x_pre[i, :, :])
            sheet['G' + str(cnt_)] = str(round(df, 2))

            area = sm.area_between_two_curves(data_x[i, :, :], data_x_pre[i, :, :])
            sheet['B' + str(cnt_)] = str(round(area, 2))

            cnt_ += 1
        workbook.save(data_root)
        return cnt_

