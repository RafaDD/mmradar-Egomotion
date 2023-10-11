import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from math import cos, sin, tanh, pi

def get_predict_path(folder):
    v = np.load(f'./result2/{folder}/radar_only.npy')
    omega = np.load(f'./result2/{folder}/imu_only.npy')[:, 5]
    v = v[:, :2]
    #print(v)
    if v[0, 0] == 0:
        if v[0, 1] != 0:
            theta = pi / 2
        else:
            theta = 0
    else:
        theta = tanh(v[0, 1] / v[0, 0])
    theta += pi / 2
    print(theta, v[0])
    pos = np.array([[0, 0]])
    for i in range(len(omega)):
        theta += omega[i] * 0.1
        rotation = np.array([[cos(theta), -sin(theta)],
                             [sin(theta), cos(theta)]])
        v_real =  rotation @ v[i]
        new_pos = pos[-1] + 0.1 * v_real
        pos = np.concatenate([pos, new_pos.reshape(1, -1)])
    return pos


def draw(folder):
    gt = loadmat(f'./dataset2/{folder}/odom_data.mat')
    gt = np.array(gt['odom_data'])

    pre = get_predict_path(folder)

    plt.clf()
    plt.plot(gt[:, 0], gt[:, 1], c='r', label='groud_truth', lw=1)
    plt.plot(pre[:,  0], pre[:, 1], c='b', label='predict', lw=1)
    plt.legend()
    plt.savefig(f'./result2/together/{folder}.png', dpi=300)

    plt.clf()
    plt.plot(gt[:, 0], gt[:, 1], c='r', label='groud_truth', lw=1)
    plt.savefig(f'./result2/{folder}/gt.png', dpi=300)

    plt.clf()
    plt.plot(pre[:, 0], pre[:, 1], c='b', label='predict', lw=1)
    plt.savefig(f'./result2/{folder}/predict.png', dpi=300)