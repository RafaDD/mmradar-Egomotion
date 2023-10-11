import torch
import numpy as np
from scipy.io import loadmat
import os
from tqdm import trange
from math import tanh, pi
from utility.test_util import convert_rel_to_44matrix
from utility.eulerangles import quat2euler

def fetch_radar(folder, eps=1e-3, iter=600, lr=1e-3):
    radar_data = loadmat(f'./dataset2/{folder}/PSO.mat')
    radar_data = np.array(radar_data['all_data_list'])[:, :-1]

    radar_indices = loadmat(f'./dataset2/{folder}/PSO_indices.mat')
    radar_indices = np.array(radar_indices['all_data_index']).reshape(-1)
    radar_indices = np.unique(radar_indices)

    imu_data = loadmat(f'./dataset2/{folder}/imu_data.mat')
    imu_data = np.array(imu_data['imu_data'])
    diff = np.diff(imu_data[:, -1])
    index = np.where(diff > 0.09)[0] + 1
    index = np.concatenate([np.zeros(1), index]).astype(np.int16)
    framed_imu = []
    for i in range(index.shape[0] - 1):
        framed_imu.append(np.mean(imu_data[index[i]:index[i+1]], axis=0))
    framed_imu = np.array(framed_imu)

    radar = []
    for i in range(1, len(radar_indices)):
        radar.append(torch.tensor(radar_data[radar_indices[i-1]: radar_indices[i]]))

    epsilon = eps
    max_iter = iter
    l_r = lr
    radar_v = [np.zeros(3)]
    frames = min(len(radar), len(framed_imu))
    for i in trange(frames, ncols=100):
        v = torch.zeros(3, requires_grad=True).type(torch.DoubleTensor)
        r_data = radar[i]
        D = radar[i].shape[0]
        for j in range(D):
            r_data[j, :3] /= torch.linalg.norm(r_data[j, :3])

        r = r_data[:, :3]
        v_r = r_data[:, -1]
        #imu_v = torch.tensor(radar_v[i] + 0.1 * framed_imu[i, :3])

        for j in range(max_iter):
            e_d = torch.sum(torch.abs(v_r + torch.matmul(r, v))) / D# + torch.dot(v - imu_v, v - imu_v) / (15 * D)
            v.retain_grad()
            e_d.backward()
            if torch.linalg.norm(v.grad) < epsilon:
                break
            v -= v.grad * l_r
        radar_v.append(v.detach().numpy())

    radar_v = np.array(radar_v)
    radar_v[:, 2] = 0
    return radar_v[1:]

def fetch_imu(folder):
    imu_data = loadmat(f'./dataset2/{folder}/imu_data.mat')
    imu_data = np.array(imu_data['imu_data'])

    diff = np.diff(imu_data[:, -1])
    index = np.where(diff > 0.09)[0] + 1
    index = np.concatenate([np.zeros(1), index]).astype(np.int16)
    framed_imu = []
    for i in range(index.shape[0] - 1):
        framed_imu.append(np.mean(imu_data[index[i]:index[i+1]], axis=0))
    framed_imu = np.array(framed_imu)

    framed_imu = framed_imu[:, 3:-1] * 0.1
    framed_imu = framed_imu
    return framed_imu

def fetch_gt(folder):
    gt = loadmat(f'./dataset2/{folder}/odom_data.mat')
    gt = np.array(gt['odom_data'])

    euler = np.zeros((gt.shape[0], 6))
    for i in range(gt.shape[0]):
        eu = quat2euler(gt[i, [6, 3, 4, 5]])
        euler[i, 3] = eu[2] 
        euler[i, 4] = eu[1] 
        euler[i, 5] = eu[0]   
    tmp = euler.copy()
    for i in range(1, gt.shape[0]):
        euler[i, 3:] = tmp[i, 3:] - tmp[i-1, 3:]
        
    euler[0, :3] = gt[0, :3].copy()
    R = np.identity(3)
    for i in range(1, gt.shape[0]):
        q = gt[i-1, 3:].copy()
        nq = np.dot(q, q)
        if nq < 1e-7:
            Rt = np.identity(3)
        else:
            q *= np.sqrt(2.0 / nq)
            q = np.outer(q, q)
            Rt = np.array([
                [1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3]],
                [q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3]],
                [q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1]]], dtype=np.float64)
        delta = gt[i, :3] - gt[i-1, :3]
        euler[i, :3] = np.linalg.inv(R) @ delta
        R = Rt
    return euler

def calc_angle(x, y):
    if x == 0:
        if y != 0:
            theta = pi / 2
        else:
            theta = 0
    else:
        theta = tanh(y / x)
    return theta


def calc_all(folder):
    v = fetch_radar(folder) * 0.1
    omega = fetch_imu(folder)
    omega_bias = np.zeros(3)
    omega_bias[2] = pi / 2
    omega[0] += omega_bias
    omega *= 180.0 / pi
    length = min(v.shape[0], omega.shape[0])
    pred = np.concatenate([v[:length], omega[:length]], axis=1)
    gt = fetch_gt(folder)

    gt_transform_t_1 = np.eye(4)
    pred_transform_t_1 = np.eye(4)
    out_gt_array = []
    out_pred_array = []
    for i in range(min(gt.shape[0], pred.shape[0])):
        pred_transform_t = convert_rel_to_44matrix(0, 0, 0, pred[i])
        abs_pred_transform = np.dot(pred_transform_t_1, pred_transform_t)

        gt_transform_t = convert_rel_to_44matrix(0, 0, 0, gt[i])
        abs_gt_transform = np.dot(gt_transform_t_1, gt_transform_t)
        
        out_gt_array.append(abs_gt_transform)
        out_pred_array.append(abs_pred_transform)
        pred_transform_t_1 = abs_pred_transform
        gt_transform_t_1 = abs_gt_transform
    np.savez(f'./{folder[5:]}/ep1.npz', pred_pos=np.array(out_pred_array), gt_pos=np.array(out_gt_array))
    print(f'{folder[5:]} finished!')