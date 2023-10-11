from filterpy.kalman.UKF import UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints
from utility.test_util import convert_rel_to_44matrix
from utility.mmradar import fetch_radar, match
from utility.eulerangles import quat2euler, euler2quat
from scipy.io import loadmat
import numpy as np
import torch
from utility.imu_gt import fetch_imu, get_q, fetch_gt
from math import pi
import os
from tqdm import trange

rnn_model = torch.load(f'./models/0926-pso-change/best.pkl').cuda(0)
q = []
a = []

def fx(x, dt):
    p_next = rnn_model(torch.Tensor(x[:7].reshape(1, 1, 7)).cuda(0)).detach().cpu().numpy().reshape(-1)# + x[:3]
    #print(p_next)
    q_next = quat2euler(x[[6, 3, 4, 5]])
    q_next += q
    q_next = euler2quat(q_next[0], q_next[1], q_next[2])[[1, 2, 3, 0]]
    v_next = x[7:] + a * 0.1
    return np.concatenate([p_next, q_next, v_next])

def hx(x):
    return x[:7]

def eval_save(pred, gt, save_dir, num=1):
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
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    np.savez(save_dir + f'/ep{num}.npz', pred_pos=np.array(out_pred_array), gt_pos=np.array(out_gt_array))


if __name__=='__main__':
    folder_out = '0926_change/'
    file_list = os.listdir(f'./dataset2/{folder_out}')
    for i in trange(len(file_list)):
        folder = folder_out + file_list[i]

        dt = 0.1
        points = MerweScaledSigmaPoints(10, alpha=.1, beta=.2, kappa=-1)
        kf = UnscentedKalmanFilter(dim_x=10, dim_z=2, dt=dt, fx=fx, hx=hx, points=points)
        kf.P *= 0.3
        z_std = 0.1
        kf.R = np.diag(np.ones(7) * z_std**2)
        kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.1**2, block_size=5)

        init_position = loadmat(f'./dataset2/{folder}/odom_data.mat')
        init_position = np.array(init_position['odom_data'])[0, 3:]
        position = np.concatenate([np.zeros(3), init_position])
        #print(position)

        imu_data, ac = fetch_imu(folder)
        imu_data[:, 2] = 0
        init_v = ac[0, :3]
        q_list = imu_data[:, :3] * 0.1
        kf.x = np.concatenate([position, init_v])

        radar_data = fetch_radar(folder)
        rotation, displace, rotation_mat = match(radar_data)
        #print(displace)

        log = [kf.x]

        for j in range(min(len(q_list), len(ac), len(displace), len(rotation))):
            q = q_list[j]
            q = q[[2, 1, 0]]
            a = ac[j]
            kf.predict()
            control = np.concatenate([displace[j] + (rotation_mat[j] @ kf.x[:3].reshape(3, 1)).reshape(-1), rotation[j]])
            #print(f'control: ')
            kf.update(control)
            log.append(kf.x)
            #print(log[-1])

        save_dir = f'./result/{file_list[i]}'

        omega, _ = fetch_imu(folder)
        omega[0, 2] -= pi / 2
        omega *= 180.0 / pi
        gt = fetch_gt(folder)

        v = np.array(log)[:, 7:] * 0.1
        # v[:, -1] = 0
        v = np.flip(v, axis=0)
        #v[:, 1] *= -1
        n = 2
        l = min(len(v), len(omega))
        pred = np.concatenate([v[:l], omega[:l]], axis=1)
        eval_save(pred, gt, save_dir, n)

        v = np.array(log)[:, :3]
        v = np.diff(v, axis=0)
        # v[:, -1] = 0
        v = np.concatenate([np.zeros(3).reshape(1, -1), v], axis=0)
        n = 3
        l = min(len(v), len(omega))
        pred = np.concatenate([v[:l], omega[:l]], axis=1)
        eval_save(pred, gt, save_dir, n)