import torch
from torch import nn
from torch.nn.modules.container import ModuleList
import torch.optim as optim
from torch.utils.data import DataLoader
from utility.rnn import RNNDataset, BiLSTM
from tqdm import trange
import numpy as np


batch_size = 1
l_r = 1e-3
epochs = 500

dataset = RNNDataset('0926_change')
dataloader = DataLoader(dataset, batch_size=batch_size)

rnn = BiLSTM(7).cuda(0)
rnn.criterion = nn.MSELoss()
rnn.optimizer = optim.Adam(rnn.parameters(), lr=l_r, betas=(0.9, 0.999), eps=1e-8)
rnn.lr_scheduler = optim.lr_scheduler.ExponentialLR(rnn.optimizer, 0.5**(1/epochs))
best_MAE = 99999999

for i in trange(epochs):
    rnn.train()
    for batch in dataloader:
        rnn.zero_grad()
        pred = rnn(batch['input'].cuda(0)).reshape(-1, 3)
        gt = batch['position'].cuda(0).reshape(-1, 3)
        loss = rnn.criterion(pred, gt)
        loss.backward(retain_graph=True)
        rnn.optimizer.step()
    if i % 20 == 0:
        rnn.eval()
        pred = []
        gt = []
        for batch in dataloader:
            pred += list(rnn(batch['input'].cuda(0)).detach().cpu().numpy().reshape(-1))
            gt += list(batch['position'].numpy().reshape(-1))
        
        pred = np.array(pred)
        gt = np.array(gt)
        MAE = np.mean(abs(pred - gt))
        print(f'Epoch: {i}, MAE: {MAE}')
        if MAE < best_MAE:
            torch.save(rnn, './models/0926-pso-change/best.pkl')
            best_MAE = MAE
    
torch.save(rnn, './models/0926-pso-change/rnn.pkl')
