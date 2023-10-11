import numpy as np
from scipy.io import loadmat
from utility.eulerangles import mat2euler, euler2quat


def fetch_radar(folder):
    radar_data = loadmat(f'./dataset2/{folder}/pc_fft.mat')
    radar_data = np.array(radar_data['all_data_list'])[:, :-1]

    radar_indices = loadmat(f'./dataset2/{folder}/pc_indices_fft.mat')
    radar_indices = np.array(radar_indices['all_data_index']).reshape(-1)
    radar_indices = np.unique(radar_indices)

    radar = []
    for i in range(1, len(radar_indices)):
        radar.append(radar_data[radar_indices[i-1]: radar_indices[i]])
    return radar
    
def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform between corresponding 3D points A->B
    Input:
      A: Nx3 numpy array of corresponding 3D points
      B: Nx3 numpy array of corresponding 3D points
    Returns:
      T: 4x4 homogeneous transformation matrix
      R: 3x3 rotation matrix
      t: 3x1 column vector
    '''

    assert len(A) == len(B)

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    W = np.dot(BB.T, AA)
    U, s, VT = np.linalg.svd(W)
    R = np.dot(U, VT)

    if np.linalg.det(R) < 0:
       VT[2,:] *= -1
       R = np.dot(U, VT)

    t = centroid_B.T - np.dot(R,centroid_A.T)

    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return T, R, t

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nx3 array of points
        dst: Nx3 array of points
    Output:
        distances: Euclidean distances (errors) of the nearest neighbor
        indecies: dst indecies of the nearest neighbor
    '''

    indecies = np.zeros(src.shape[0], dtype=np.int32)
    distances = np.zeros(src.shape[0])
    for i, s in enumerate(src):
        min_dist = np.inf
        for j, d in enumerate(dst):
            dist = np.linalg.norm(s-d)
            if dist < min_dist:
                min_dist = dist
                indecies[i] = j
                distances[i] = dist    
    return distances, indecies

def icp(A, B, init_pose=None, max_iterations=1000, tolerance=0.001):
    '''
    The Iterative Closest Point method
    Input:
        A: Nx3 numpy array of source 3D points
        B: Nx3 numpy array of destination 3D point
        init_pose: 4x4 homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation
        distances: Euclidean distances (errors) of the nearest neighbor
    '''
    src = np.ones((4,A.shape[0]))
    dst = np.ones((4,B.shape[0]))
    src[0:3,:] = np.copy(A.T)
    dst[0:3,:] = np.copy(B.T)

    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        distances, indices = nearest_neighbor(src[0:3,:].T, dst[0:3,:].T)
        T,_,_ = best_fit_transform(src[0:3,:].T, dst[0:3,indices].T)
        src = np.dot(T, src)
        mean_error = np.sum(distances) / distances.size
        if abs(prev_error-mean_error) < tolerance:
            break
        prev_error = mean_error

    T,_,_ = best_fit_transform(A, src[0:3,:].T)

    return T, distances


def match(radar):
    rotation = []
    displace = []
    R_raw = []
    for i in range(len(radar) - 1):
        H, dis = icp(radar[i][:, :-1], radar[i+1][:, :-1])
        eu = np.array(mat2euler(H[:3, :3]))
        R_raw.append(H[:3, :3])
        rotation.append(euler2quat(eu[0], eu[1], eu[2])[[1, 2, 3, 0]])
        displace.append(H[:3, -1])
    return rotation, displace, R_raw
