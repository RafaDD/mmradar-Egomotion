import numpy as np
from scipy.io import loadmat
from utility.eulerangles import euler2quat, quat2euler
from math import pi


def fetch_imu(folder):
    imu_data = loadmat(f'D:/lab_work/milliego/baseline_1/dataset2/{folder}/imu_data.mat')
    imu_data = np.array(imu_data['imu_data'])

    diff = np.diff(imu_data[:, -1])
    index = np.where(diff > 0.09)[0] + 1
    index = np.concatenate([np.zeros(1), index]).astype(np.int16)
    framed_imu = []
    for i in range(index.shape[0] - 1):
        framed_imu.append(np.mean(imu_data[index[i]:index[i+1]], axis=0))
    framed_imu = np.array(framed_imu)

    ac = framed_imu[:, :3]
    framed_imu = framed_imu[:, 3:-1] * 0.1
    framed_imu[0, 2] += pi / 2
    return framed_imu, ac

def get_q(imu_data):
    q_list = []
    for i in range(len(imu_data)):
        q_list.append(euler2quat(imu_data[i, 2], imu_data[i, 1], imu_data[i, 0]))
    q_list = np.array(q_list)
    return q_list

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