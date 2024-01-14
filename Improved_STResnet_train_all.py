import torch
from Improved_STResnet import *
import pandas as pd
import numpy as np
from Improved_STResnet_dataset import *
import matplotlib.pyplot as plt
from Metric import *
from sklearn.model_selection import KFold
#datatype set to double
torch.set_default_tensor_type(torch.DoubleTensor)
path = "NN/ISTDNN_all/"
#train device
device = torch.device("cuda")

def all_train(epoch_num,learningrate,batchsize,ev_num,day_num,month_num,year_num,SW,random_state,device):
    #load data
    inputpath="pm+gwr_day+time+ev.csv"
    df=pd.read_csv(inputpath)
    #load data
    train_dataset =STDNN_Dataset(df)
    #loader
    train_loader = DataLoader(train_dataset,batch_size= batchsize,shuffle=True,num_workers=0,drop_last=False)

    #create net
    STDNN = STDNnet(train_dataset.train_x_len,ev_num=ev_num,day=day_num,month=month_num,year=year_num)
    STDNN=STDNN.to(device)

    #loss function
    loss_fn = StationweightedLoss()
    loss_fn = loss_fn.to(device)

    #optimizer
    params = [p for p in STDNN.parameters() if p.requires_grad]
    lr= learningrate
    optimizer=torch.optim.Adam(params,lr= lr)
    #train
    epoch = epoch_num
    train_r2_list = []
    train_rmse_list = []
    train_mae_list = []
    max_train_r2 = 0
    max_train_rmse = 0
    max_train_mae = 0
    Epoch_list=[]
    Truevalues=[]
    Estimates=[]

    for i in range(epoch+1):
        STDNN.train()
        for data in train_loader:
            factors, targets, evs, years, months, days,weight_dem = data
            factors = factors.to(device)
            targets = targets.to(device)
            evs = evs.to(device)
            years = years.to(device)
            months = months.to(device)
            days = days.to(device)
            weight_dem = weight_dem.to(device)

            for target in targets.cpu().numpy():
                Truevalues.append(target)
             #output
            outputs = STDNN(factors,evs,days,months,years)

            for output in outputs.data.cpu().numpy():
                Estimates.append(output)

            #loss compute
            loss = loss_fn(outputs,targets,weight_dem,super_weight=SW)
            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # visualization
        Epoch_list.append(i)
        train_r2_list.append(R2(Estimates, Truevalues))
        train_rmse_list.append(rmse(Estimates, Truevalues))
        train_mae_list.append(mae(Estimates,Truevalues))
        if train_r2_list[-1] >= max_train_r2:
            max_train_r2 = train_r2_list[-1]
            max_train_rmse = train_rmse_list[-1]
            max_train_mae = train_mae_list[-1]
            max_epoch_train = i
            if 940>i>=800:
                torch.save(STDNN, path + "ISTDNN" + str(SW) + "_sample_aod" + ".pt")

        if Epoch_list[-1] % 10 == 0:
            print("Epoch:{}, train R2:{},RMSE:{}, MAE:{}".format(Epoch_list[-1], train_r2_list[-1], train_rmse_list[-1],train_mae_list[-1]))
        Truevalues.clear()
        Estimates.clear()
    print("epoch:{}, r2:{}, rmse:{}, mae:{} ".format(max_epoch_train,max_train_r2,max_train_rmse,max_train_mae))





if __name__ == '__main__':
    SW=2
    print(SW)
    #sample_train(1000,0.005,640,5,15,6,2,SW,random_state=345,device=device)
    all_train(1000,0.005,6400,5,15,6,2,SW,random_state=345,device=device)


