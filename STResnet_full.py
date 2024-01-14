import torch
from STResnet import *
import pandas as pd
import numpy as np
from Improved_STResnet_dataset import *
import matplotlib.pyplot as plt
from Metric import *
torch.set_default_tensor_type(torch.DoubleTensor)
device='cuda'
inputpath="NN/STDNN/STDNN_normal/"
outputpath="NN/STDNN_csv/"
def loaddata(inputpath,outputpath):

    datapath="pm+gwr_day+time+ev_n.csv"
    df=pd.read_csv(datapath)
    data_1 = df.loc[df['code'].isin(['1002A','1058A'])]
    data_2 = df.loc[df['code']=='1059A']
    data_3 = df.loc[df['code'].isin(['1009A','1061A','1064A'])]
    data_4 = df.loc[df['code']=='1035A']
    data_5 = df.loc[df['code']=='1065A']
    data_6 = df.loc[df['code'].isin(['1062A','lo63A'])]

    #dataload
    dataset_1 = STDNN_Dataset(data_1)
    dataset_2 = STDNN_Dataset(data_2)
    dataset_3 = STDNN_Dataset(data_3)
    dataset_4 = STDNN_Dataset(data_4)
    dataset_5 = STDNN_Dataset(data_5)
    dataset_6 = STDNN_Dataset(data_6)

    #dataloader
    dataloaders=[]
    dataloader_1=DataLoader(dataset_1, batch_size=640, shuffle=True, num_workers=0, drop_last=False)
    dataloader_2=DataLoader(dataset_2, batch_size=640, shuffle=True, num_workers=0, drop_last=False)
    dataloader_3=DataLoader(dataset_3, batch_size=640, shuffle=True, num_workers=0, drop_last=False)
    dataloader_4=DataLoader(dataset_4, batch_size=640, shuffle=True, num_workers=0, drop_last=False)
    dataloader_5=DataLoader(dataset_5, batch_size=640, shuffle=True, num_workers=0, drop_last=False)
    dataloader_6=DataLoader(dataset_6, batch_size=640, shuffle=True, num_workers=0, drop_last=False)
    dataloaders.append(dataloader_1)
    dataloaders.append(dataloader_2)
    dataloaders.append(dataloader_3)
    dataloaders.append(dataloader_4)
    dataloaders.append(dataloader_5)
    dataloaders.append(dataloader_6)
    #model_load
    models=[]
    model_1=torch.load(inputpath+"STDNN_normalSTDNN1_station.pt")
    model_2=torch.load(inputpath+"STDNN_normalSTDNN2_station.pt")
    model_3=torch.load(inputpath+"STDNN_normalSTDNN3_station.pt")
    model_4=torch.load(inputpath+"STDNN_normalSTDNN4_station.pt")
    model_5=torch.load(inputpath+"STDNN_normalSTDNN5_station.pt")
    model_6=torch.load(inputpath+"STDNN_normalSTDNN6_station.pt")
    models.append(model_1)
    models.append(model_2)
    models.append(model_3)
    models.append(model_4)
    models.append(model_5)
    models.append(model_6)

    #datasave
    estimates=[]
    trueValues=[]
    metrics=[]
    for i in range(6):
        for data in dataloaders[i]:
            factors, targets, evs, years, months, days = data
            factors = factors.to(device)
            targets = targets.to(device)
            evs = evs.to(device)
            years = years.to(device)
            months = months.to(device)
            days = days.to(device)
            for target in targets.cpu().numpy():
                trueValues.append(target)
            # output
            outputs = models[i](factors, evs, days, months, years)
            for output in outputs.data.cpu().numpy():
                estimates.append(output)
    result=pd.DataFrame([trueValues, estimates])
    result=result.T
    result.columns=['true','estimate']
    result.to_csv(outputpath+'result_dem_n.csv',index=False)
    metrics.append(R2(trueValues, estimates))
    metrics.append(rmse(trueValues, estimates))
    metrics.append(mae(trueValues, estimates))
    result_metrics=pd.DataFrame(metrics)
    result_metrics=result_metrics.T
    result_metrics.columns=['R2','RMSE','MAE']
    result_metrics.to_csv(outputpath+"result_metrics_dem_n.csv",index=False)

loaddata(inputpath,outputpath)