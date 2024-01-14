#预测没有站点处的PM2.5浓度
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class STDNN_Dataset(Dataset):
    def __init__(self, df):
        super().__init__()
        #self.x = torch.from_numpy(np.array(df.loc[:,["AOD","NDVI","RH","TS","PS","PBLH","ROAD","FACT","DEM","EV1_N","EV2_N","EV3_N","EV4_N","EV5_N","EV6_N","EV7_N","EV8_N","YEAR_N","MONTH_N","DAY_N"]]))
        #self.x = torch.from_numpy(np.array(df.loc[:,["NDVI","RH","TS","PS","PBLH","ROAD","FACT","DEM"]]))
        self.x = torch.from_numpy(np.array(df.loc[:,["AOD","NDVI","RH","TS","PS","PBLH","ROAD","FACT","DEM"]]))
        #self.ev = torch.from_numpy(np.array(df.loc[:,["EV1_N","EV2_N","EV3_N","EV4_N","EV5_N","EV6_N","EV7_N","EV8_N"]]))
        self.ev = torch.from_numpy(np.array(df.loc[:,["EV1","EV2","EV3","EV4","EV5","EV6","EV7","EV8"]]))
        inputpath = "pm+gwr_day+time+ev_n.csv"
        df1 = pd.read_csv(inputpath)
        self.weight_dem = torch.from_numpy(np.array(df1.loc[:,"DEM"]))
        #self.id = torch.from_numpy(np.array(df.loc[:,"POINTID"]))
        self.year = self.embedding_year(df)
        self.month = self.embedding_month(df)
        self.day = self.embedding_day(df)
        self.len = len(df)
        self.train_x_len=8

    def __getitem__(self, index):
        return self.x[index],self.ev[index],self.year[index],self.month[index],self.day[index]
    def embedding_year(self,df):
        year = np.array(df.loc[:,"YEAR"])-2015
        year_onehot= torch.from_numpy(np.eye(4,4)[year.reshape(year.shape[0])])

        return year_onehot

    def embedding_month(self,df):
        month = np.array(df.loc[:,"MONTH"])-1
        month_onehot= torch.from_numpy(np.eye(12,12)[month.reshape(month.shape[0])])
        return month_onehot
    def embedding_day(self,df):
        day = np.array(df.loc[:,"DAY"])-1
        day_onehot= torch.from_numpy(np.eye(31,31)[day.reshape(day.shape[0])])
        return day_onehot
    def __len__(self):
        return self.len

