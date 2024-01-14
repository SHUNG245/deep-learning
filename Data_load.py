import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co, random_split

class MyDataset(Dataset):

    def __init__(self,inputpath):
        super().__init__()
        data_csv = pd.read_csv(inputpath)
        # dim=[2598,9]
        #self.x = torch.from_numpy(np.array(data_csv.values[:,7:]).astype(float))
        self.x = torch.from_numpy(np.array(data_csv.values[:,5:-1]).astype(float))
        self.train_x_size = self.x.size()[1]
        #self.y = torch.from_numpy(np.array(data_csv.values[:,4]).astype(float))
        self.y = torch.from_numpy(np.array(data_csv.values[:,-1]).astype(float))
        self.len = len(data_csv)

    def __getitem__(self, index):
        return self.x[index],self.y[index]

    def __len__(self):
        return self.len



