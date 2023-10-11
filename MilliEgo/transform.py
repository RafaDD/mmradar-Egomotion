import numpy as np
from scipy.io import loadmat
import math
import os
from sklearn.preprocessing import normalize
from eulerangles import quat2euler, euler2mat, euler2quat
import multiprocessing as mp
from numba import jit

@jit
def expand(image):
    output = np.zeros((64, 256))
    for i in range(32):
        for j in range(128):
            if image[i, j] != 0:
                output[2*i, 2*j] = max(output[2*i, 2*j], image[i, j])
                output[2*i+1, 2*j] = max(output[2*i+1, 2*j], image[i, j])
                output[2*i, 2*j+1] = max(output[2*i, 2*j+1], image[i, j])
                output[2*i+1, 2*j+1] = max(output[2*i+1, 2*j+1], image[i, j])
    return output

@jit
def smooth(image):
    output = np.zeros((64, 256))
    for i in range(64):
        for j in range(256):
            cnt = 0
            for a in range(-1, 2):
                for b in range(-1, 2):
                    if i+a < 64 and i+a >= 0 and j+b < 256 and j+b >= 0:
                        output[i, j] += image[i+a, j+b]
                        cnt += 1
            output[i, j] /= (cnt - 2)
    return output

def fetch_data(folder):
    imu_data = loadmat(f'{folder}/imu_data.mat')
    imu_data = np.array(imu_data['imu_data'])

    radar_data = loadmat(f'{folder}/PSO.mat')
    radar_data = np.array(radar_data['all_data_list'])

    gt = loadmat(f'{folder}/odom_data.mat')
    gt = np.array(gt['odom_data'])

    radar_indices = loadmat(f'{folder}/PSO_indices.mat')
    radar_indices = np.array(radar_indices['all_data_index']).reshape(-1)
    
    return imu_data, radar_data, gt, radar_indices

@jit
def fetch_depth_image(radar_data, radar_indices, flip=False):
    total_image = []
    x_reso = math.pi / 128
    y_reso = math.pi * 2 / (4 * 32)
    for i in range(1, len(radar_indices)):
        image = np.zeros((32, 128))
        r_list = []
        x = []
        y = []
        for j in range(radar_indices[i-1], radar_indices[i]):
            r = math.sqrt(np.sum(radar_data[j, :3] ** 2))
            r_list.append(r)
            r_list[-1] = max(r_list[-1], 0.01)
            x.append(int(math.atan2(radar_data[j, 1], radar_data[j, 0]) / x_reso))
            x[-1] = min(x[-1], 127)
            y.append(int((math.asin(radar_data[j, 2] / r_list[-1]) + math.pi / 4) / y_reso))
            y[-1] = min(y[-1], 31)
        if len(r_list) > 0:
            farthest = np.max(r_list)
            r_list = np.abs(np.array(r_list) - farthest)
            for j in range(len(r_list)):
                if r_list[j] > image[y[j], x[j]]:
                    image[y[j], x[j]] = r_list[j]
        image = expand(image)
        image = smooth(image)
        image = smooth(image)
        image = smooth(image)
        image = np.array(normalize(image.reshape(1, -1), norm='max')).reshape(64, 256) - 0.03
        total_image.append(image)
    if flip:
        total_image = np.flipud(np.array(total_image))
    total_image = np.array(total_image).reshape(-1, 1, 64, 256, 1)
    return total_image

def fetch_imu_data(imu_data, flip=False):
    imu_range = []
    for i in range(imu_data.shape[0] - 1):
        imu_range.append(imu_data[i+1, -1] - imu_data[i, -1])

    imu_length = []
    min_frame = 100
    for i in range(len(imu_range)):
        if imu_range[i] > 0.05:
            imu_length.append(i)
    for i in range(len(imu_length) - 1):
        min_frame = min(min_frame, imu_length[i+1] - imu_length[i])
    
    min_frame = min(min_frame, 19)
    imu_data_save = []
    imu_data_save.append(imu_data[:min_frame, :-1])
    for i in range(len(imu_length) - 1):
        imu_data_save.append(imu_data[imu_length[i]+1:imu_length[i]+1+min_frame, :-1])
    if flip:
        imu_data_save = np.flipud(np.array(imu_data_save).reshape(-1))
    imu_data_save = np.array(imu_data_save).reshape(-1, min_frame, 6)
    return imu_data_save, min_frame

@jit
def gt_q2euler(gt, flip=False):
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
    if flip:
        euler[:, :3] *= -1
        euler = np.flipud(euler)
    return euler


def transform_npz(num):
    folder = './data_0926_change/' + num
    imu_data, radar_data, gt, radar_indices = fetch_data(folder)
    total_image = fetch_depth_image(radar_data, radar_indices, flip=True)
    imu_data_save, min_frame = fetch_imu_data(imu_data, flip=True)
    gt = gt_q2euler(gt, flip=True)

    num = int(num)
    print(f'{num}, {min_frame}')
    gt_save = gt.reshape(1, -1, 6)
    print(gt_save.shape)
    np.savez(f'./0926_pso_change/test/data{num}.npz', mm=total_image, imu=imu_data_save, gt=gt_save)
    

if __name__ == '__main__':
    folders = os.listdir('./data_0926_change')
    pool = mp.Pool(15)
    pool.map(transform_npz, folders)
    pool.close()
    pool.join()
