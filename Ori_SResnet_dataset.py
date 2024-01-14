import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class Ori_Resnet_dataset(Dataset):
    def __init__(self, df):
        super().__init__()
        #self.x = torch.from_numpy(np.array(df.loc[:,["AOD","NDVI","RH","TS","PS","PBLH","ROAD","FACT","DEM","EV1_N","EV2_N","EV3_N","EV4_N","EV5_N","EV6_N","EV7_N","EV8_N"]]))

        self.x = torch.from_numpy(np.array(df.loc[:,["AOD","NDVI","RH","TS","PS","PBLH","ROAD","FACT","DEM"]]))
        #self.ev = torch.from_numpy(np.array(df.loc[:,["EV1_N","EV2_N","EV3_N","EV4_N","EV5_N","EV6_N","EV7_N","EV8_N"]]))
        #self.ev = torch.from_numpy(np.array(df.loc[:,["EV1","EV2","EV3","EV4","EV5","EV6","EV7","EV8"]]))
        self.ev = torch.from_numpy(np.array(df.loc[:,["long","lat"]]))
        self.dem = df.loc[:,"DEM"]
        self.dem_weight = df.loc[:,"DEM"]/ max(df.loc[:,"DEM"])
        self.y = torch.from_numpy(np.array(df.loc[:,"PM"]))
        #self.time=torch.from_numpy(np.array(df.loc[:,["YEAR_N","MONTH_N","DAY_N"]]))
        self.time=torch.from_numpy(np.array(df.loc[:,["YEAR","MONTH","DAY"]]))
        self.len = len(df)
        self.time_size=3
        self.train_x_len=9

    def __getitem__(self, index):
        return self.x[index], self.y[index],self.ev[index],self.time[index]

    def __len__(self):
        return self.len


