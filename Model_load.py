#把站点作为输入得到拟合的结果
import os
import torch
import pandas as pd
import numpy as np
from Ori_SResnet_dataset import *
from All_region_loade_dataset import *
import matplotlib.pyplot as plt
from Metric import *
torch.set_default_tensor_type(torch.DoubleTensor)
device='cuda'
inputpath="pm+gwr_day+time+ev.csv"
netPath="NN/ISTDNN_all/ISTDNN2_sample.pt"
#netPath="NN/Ori-STResNet/Ori-STResNet_all.pt"
outpath="NN/Ori-STResNet_csv"
def loadData(filename):
    df = pd.read_csv(filename)
    dataset = Ori_Resnet_dataset(df)
    dataloader = DataLoader(dataset,batch_size=640, shuffle=False, num_workers=0, drop_last=False)
    #model_load
    model=torch.load(netPath)
    #datasave
    model.eval()
    estimates=[]
    with torch.no_grad():
        for data in dataloader:
            factors, targets, evs,times= data
            factors = factors.to(device)
            targets = targets.to(device)
            evs = evs.to(device)
            times = times.to(device)
            # output
            outputs = model(factors, evs, times)
            for output in outputs.cpu().numpy():
                estimates.append(output)
    df['estimates']=estimates
    df.to_csv(outpath+'/'+'ISTResNet.csv',index=False)

#添加ISTDNN的数据
def loadData1(filename):
    df = pd.read_csv(filename)
    dataset = STDNN_Dataset(df)
    dataloader = DataLoader(dataset, batch_size=df.shape[0], shuffle=False, num_workers=0, drop_last=False)
    #model_load
    model=torch.load(netPath)
    #datasave
    estimates=[]
    model.eval()

    with torch.no_grad():
        for data in dataloader:
            factors, evs, years, months, days = data
            factors = factors.to(device)
            evs = evs.to(device)
            years = years.to(device)
            months = months.to(device)
            days = days.to(device)
            # output
            outputs = model(factors, evs, days, months, years)
            for output in outputs.data.cpu().numpy():
                estimates.append(output)

        df['estimates'] = estimates
        df.to_csv(outpath + '/' + 'ISTResNet.csv', index=False)
#loadData(inputpath)
loadData1(inputpath)