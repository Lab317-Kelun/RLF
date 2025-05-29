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

Method = "WASQF"
if Method == "REF":
    from Algorithms.Learn_RSP_CLF_REF import LearnClf
elif Method == "NEUM":
    from Algorithms.Learn_RSP_CLF_NEUM import LearnClf
elif Method == "WASQF":
    from Algorithms.Learn_RSP_CLF_WASQF import LearnClf

np.random.seed(0)
torch.manual_seed(0)

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

    # data = np.load('Datasets/data_set_1.npz')
    # x_set = data['data_x']
    # dot_x_set = data['data_y']
    # t_set = data['data_t']
    return x_set, dot_x_set, t_set

cnt_ = 3  # 计数
t_sum = 0  # 总训练时间
# All_Type = ['Angle','BendedLine','CShape','DoubleBendedLine','GShape','heee','JShape','JShape_2','Khamesh','Leaf_1','Leaf_2','Line','LShape','NShape','PShape','RShape','Saeghe','Sharpc','Sine','Snake','Spoon','Sshape','Trapezoid','WShape','Worm','Zshape']
All_Type = ['BendedLine','DoubleBendedLine','JShape_2','Leaf_1','Leaf_2','Snake']
for Type in All_Type:
    print(Type)
    data = getattr(lasa.DataSet, Type)
    demos = data.demos

    manually_design_set_neum = construct_demonstration_set(demos, start=20, end=-1, gap=5)
    data_x, data_y, data_t = manually_design_set_neum
    clf_learner = LearnClf((data_x, data_y, data_t))

    save_path = 'ClfParameters/Clf_parameter_for_' + Type + '_' + Method + '.pth'
    clf_learner.load_state_dict(torch.load(save_path))
    start_time = time.time()

    clf_learner.train_energy(save_path)
    clf_learner.train_ods(save_path)
    # clf_learner.train_all(save_path)

    end_time = time.time()
    execution_time = end_time - start_time
    t_sum += execution_time
    print("Execution time: ", execution_time, "seconds")

    manually_design_set_sds = construct_demonstration_set(demos, start=20, end=-1, gap=5)
    sds_learner = LearnSds(manually_design_set_sds, clf_learner, Type)

    save_path = 'Result_fig/RSP_' + Type + '_' + Method
    cnt_ = sds_learner.simutation(save_path, Type, cnt_, execution_time, Method, False)

print(t_sum/26)  # 平均时间