from scipy.io import loadmat
import numpy as np
from torch import nn
import torch
from torch.utils.data import Dataset
import os


class BiLSTM(nn.Module):
    def __init__(self, input_dim):
        super(BiLSTM, self).__init__()
        self.fwlstm = nn.LSTM(input_dim, 256, num_layers=2, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,3)
        )
        
    def forward(self, X):
        x = self.fwlstm(X)
        x = x[0].reshape(-1, 512)
        res = self.fc(x)
        return res


class RNNDataset(Dataset):
    def __init__(self, name):
        super(RNNDataset, self).__init__()

        dir = f'./dataset2/{name}/'
        files = os.listdir(dir)
        self.gt = []
        self.y = []
        for i in files:
            file_dir = dir + i
            gt = np.array(loadmat(file_dir + '/odom_data.mat')['odom_data'])
            y = np.array(loadmat(file_dir + '/odom_data.mat')['odom_data'])
            for j in range(1, gt.shape[0]):
                y[j, :3] = gt[j, :3] - gt[j-1, :3]
            y[0, :3] = 0
            gt[:, :3] -= gt[0, :3]
            self.gt.append(gt)
            self.y.append(y)
    
    def __getitem__(self, index):
        return_dict = {
            'input': torch.FloatTensor(self.gt[index][:-1, :7]),
            'position': torch.FloatTensor(self.gt[index][1:, :3]),
        }
        return return_dict
    
    def __len__(self):
        return len(self.gt)
    