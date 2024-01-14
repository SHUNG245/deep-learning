import torch
from Ori_SResnet_dataset import *
import pandas as pd
import numpy as np
from Ori_SResnet_model import *
import matplotlib.pyplot as plt
from Metric import *
from sklearn.model_selection import KFold
#datatype set to double
torch.set_default_tensor_type(torch.DoubleTensor)

#train device
device = torch.device("cuda")
outModelPath="NN/Ori-STResNet"
outCsvPath="NN/Ori-STResNet_csv"
def all_train(epoch_num, learningrate, batchsize, device):
    #load data
    inputpath="pm+gwr_day+time+ev.csv"
    df=pd.read_csv(inputpath)
    #load data
    train_dataset =Ori_Resnet_dataset(df)
    #loader
    train_loader = DataLoader(train_dataset,batch_size= batchsize,shuffle=True,num_workers=0,drop_last=False)
    #create net
    Oris = Ori_Resnet(train_dataset.train_x_len)
    Oris = Oris.to(device)

    # loss function
    loss_fn = nn.MSELoss()
    loss_fn = loss_fn.to(device)
    # optimizer
    params = [p for p in Oris.parameters() if p.requires_grad]
    print(Oris.parameters())
    lr = learningrate
    optimizer = torch.optim.Adam(params, lr=lr)
    # train
    epoch = epoch_num
    Epoch_list = []
    Truevalues = []
    Estimates = []
    lon=[]
    lat=[]
    year=[]
    month=[]
    day=[]
    train_r2_list=[]
    train_rmse_list=[]
    train_mae_list=[]

    #final
    ESTIMATES=[]
    OBSERVATION=[]
    LON=[]
    LAT=[]
    YEAR=[]
    MONTH=[]
    DAY=[]

    for i in range(epoch+1):
        Oris.train()
        for data in train_loader:
            factors, targets, evs, times = data
            factors = factors.to(device)
            targets = targets.to(device)
            evs = evs.to(device)
            times = times.to(device)
            for target in targets.cpu().numpy():
                Truevalues.append(target)
            for ev in evs.cpu().numpy():
                lon.append(ev[0])
                lon.append(ev[1])
            for time in times.cpu().numpy():
                year.append(time[0])
                month.append(time[1])
                day.append(time[2])
            #output
            outputs = Oris(factors,evs,times)

            for output in outputs.data.cpu().numpy():
                Estimates.append(output)

            # loss compute
            loss = loss_fn(input=outputs, target=targets)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # visualization
        Epoch_list.append(i)
        train_r2_list.append(R2(Estimates, Truevalues))
        train_rmse_list.append(rmse(Estimates, Truevalues))
        train_mae_list.append(mae(Estimates, Truevalues))
        if Epoch_list[-1] % 10 == 0:
            print("Epoch:{}, train R2:{},RMSE:{}, MAE:{}".format(Epoch_list[-1], train_r2_list[-1], train_rmse_list[-1],
                                                                 train_mae_list[-1]))
        if i == 950:
            for e,o,long,lati, d, m,y in zip(Estimates,Truevalues,lon,lat,day,month,year):
                ESTIMATES.append(e)
                OBSERVATION.append(o)
                LON.append(long)
                LAT.append(lati)
                YEAR.append(y)
                MONTH.append(m)
                DAY.append(d)
            torch.save(Oris,outModelPath+"/Ori-STResNet_all.pt")
        Truevalues.clear()
        Estimates.clear()
    df_e = pd.DataFrame(ESTIMATES)
    df_o = pd.DataFrame(OBSERVATION)
    df_lon = pd.DataFrame(LON)
    df_lat = pd.DataFrame(LAT)
    df_year = pd.DataFrame(YEAR)
    df_month = pd.DataFrame(MONTH)
    df_day = pd.DataFrame(DAY)
    df = pd.concat([df_e,df_o,df_lon,df_lat,df_year,df_month,df_day],axis=1)
    df.to_csv(outCsvPath+'/Ori-STResNet.csv',index=False)
all_train(1000,0.05,640,device)