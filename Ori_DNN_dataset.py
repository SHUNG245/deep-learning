import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class MyDataset_initial(Dataset):
    def __init__(self, df):
        super().__init__()
        self.x = torch.from_numpy(np.array(df.loc[:,["AOD","NDVI","RH","TS","PS","PBLH","ROAD","FACT","DEM","EV1_N","EV2_N","EV3_N","EV4_N","EV5_N","EV6_N","EV7_N","EV8_N","YEAR_N","MONTH_N","DAY_N"]]))
        #self.x = torch.from_numpy(np.array(df.loc[:,["AOD","NDVI","RH","TS","PS","PBLH","ROAD","FACT","DEM","EV1_N","EV2_N","EV3_N","EV4_N","EV5_N","EV6_N","EV7_N","EV8_N"]]))
        #self.x = torch.from_numpy(np.array(df.loc[:,["AOD","NDVI","RH","TS","PS","PBLH","ROAD","FACT","DEM"]]))
        self.y = torch.from_numpy(np.array(df.loc[:,"PM"]))
        self.len = len(df)
        self.train_x_len=20

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


