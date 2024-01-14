import torch
from Ori_DNN_model import *
import pandas as pd
import numpy as np
from Ori_DNN_dataset import *
import matplotlib.pyplot as plt
from Metric import *
from sklearn.model_selection import KFold
#datatype set to double
torch.set_default_tensor_type(torch.DoubleTensor)
# train device
device = torch.device("cuda")
def sample_train(epoch_num,learningrate,batchsize,device):


    #load data
    inputpath="pm+gwr_day+time+ev_n.csv"
    df=pd.read_csv(inputpath)
    #划分10折训练集和测试集

    train_r2_list=[]
    train_rmse_list=[]
    test_r2_list=[]
    test_rmse_list=[]
    test_mae_list=[]
    max_train_r2=0
    max_train_rmse=0
    max_train={}
    max_test_r2=0
    max_test_rmse=0
    max_test_mae=0
    max_test={}
    max_epoch_train=0
    max_epoch_test=0

    final_train_r2=0
    final_train_rmse=0
    final_test_r2=0
    final_test_rmse=0
    final_test_mae=0

    final_max_train_r2=0
    final_max_train_rmse=0
    final_max_test_r2=0
    final_max_test_rmse=0
    final_max_test_mae=0

    n=0
    #kf=KFold(n_splits=10,shuffle=False)#station cv
    kf=KFold(n_splits=10,shuffle=True,random_state=345) #sample cv
    for train,test in kf.split(df):
        n+=1
        train_data=df.iloc[train]
        test_data=df.iloc[test]
        # load data
        train_dataset=MyDataset_initial(train_data)
        test_dataset=MyDataset_initial(test_data)
        # loader
        train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=0, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=0, drop_last=False)
        # create net
        mynet = MyNet_ini(train_x_size=train_dataset.train_x_len)
        mynet = mynet.to(device)

        # loss function
        loss_fn = nn.MSELoss()
        loss_fn = loss_fn.to(device)

        # optimizer
        lr = learningrate
        optimizer = torch.optim.Adam(mynet.parameters(), lr=lr)
        #train
        epoch = epoch_num
        Epoch_list=[]
        Truevalues=[]
        Estimates=[]
        for i in range(epoch):
            mynet.train()
            for data in train_loader:
                factors, targets= data
                factors=factors.to(device)
                targets = targets.to(device)

                for target in targets.cpu().numpy():
                    Truevalues.append(target)
                outputs = mynet(factors)
                for output in outputs.data.cpu().numpy():
                    Estimates.append(output)

                loss= loss_fn(input= outputs,target=targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #visualization
            Epoch_list.append(i)
            train_r2_list.append(R2(Estimates,Truevalues))
            train_rmse_list.append(rmse(Estimates,Truevalues))
            if train_r2_list[-1] >= max_train_r2:
                max_train_r2= train_r2_list[-1]
                max_train_rmse = train_rmse_list[-1]
                max_epoch_train = i

            if Epoch_list[-1]%10 ==0:
                print("Epoch:{}, train R2:{},RMSE:{}".format(Epoch_list[-1],train_r2_list[-1],train_rmse_list[-1]))
            Truevalues.clear()
            Estimates.clear()

            #test
            mynet.eval()
            with torch.no_grad():
                for data in test_loader:
                    factors, targets = data
                    factors = factors.to(device)
                    targets = targets.to(device)

                    for target in targets.cpu().numpy():
                        Truevalues.append(target)

                    outputs=mynet(factors)
                    for output in outputs.cpu().numpy():
                        Estimates.append(output)
                test_r2_list.append(R2(Estimates,Truevalues))
                test_rmse_list.append(rmse(Estimates,Truevalues))
                test_mae_list.append(mae(Estimates,Truevalues))

                if test_r2_list[-1] >= max_test_r2:
                    max_test_r2 = test_r2_list[-1]
                    max_test_rmse = test_rmse_list[-1]
                    max_test_mae = test_mae_list[-1]
                    max_epoch_test = i

                if Epoch_list[-1]%10==0:
                    print("epoch: {}, test R2:{},RMSE:{}".format(Epoch_list[-1],test_r2_list[-1], test_rmse_list[-1]),test_mae_list[-1])

                Truevalues.clear()
                Estimates.clear()

        max_train["{}:epoch{}:train_max_r2".format(n,max_epoch_train)] = max_train_r2
        max_train["{}:epoch{}:train_max_rmse".format(n,max_epoch_train)] = max_train_rmse
        max_test["{}:epoch{}:test_max_r2".format(n,max_epoch_test)] = max_test_r2
        max_test["{}:epoch{}:test_max_rmse".format(n,max_epoch_test)] = max_test_rmse
        max_test["{}:epoch{}:test_max_mae".format(n,max_epoch_test)] = max_test_mae

        final_max_train_r2 += max_train_r2
        final_max_train_rmse += max_train_rmse
        final_max_test_r2 += max_test_r2
        final_max_test_rmse += max_test_rmse
        final_max_test_mae += max_test_mae

        max_train_r2 = 0
        max_train_rmse = 0
        max_test_r2 = 0
        max_test_rmse = 0
        max_test_mae = 0

        final_train_r2=final_train_r2+train_r2_list[-1]
        final_train_rmse=final_train_rmse+train_rmse_list[-1]
        final_test_r2=final_test_r2+test_r2_list[-1]
        final_test_rmse=final_test_rmse+test_rmse_list[-1]
        final_test_mae=final_test_mae+test_mae_list[-1]

    final_train_r2=final_train_r2/10
    final_train_rmse=final_train_rmse/10
    final_test_r2=final_test_r2/10
    final_test_rmse=final_test_rmse/10
    final_test_mae=final_test_mae/10

    final_max_train_r2=final_max_train_r2/10
    final_max_train_rmse=final_max_train_rmse/10
    final_max_test_r2=final_max_test_r2/10
    final_max_test_rmse=final_max_test_rmse/10
    final_max_test_mae=final_max_test_mae/10

    print("10 folds train R2:{},    RMSE:{}".format(final_train_r2,final_train_rmse))
    print("10 folds CV R2:{},    RMSE:{},   MAE: {}".format(final_test_r2,final_test_rmse,final_test_mae))

    print("10 folds max_train R2:{},    RMSE:{}".format(final_max_train_r2,final_max_train_rmse))
    print("10 folds max_CV R2:{},    RMSE:{}, MAE:{}".format(final_max_test_r2,final_max_test_rmse,final_max_test_mae))

    for k,v in max_train.items():
        print("{}:  {}".format(k,v))

    for k,v in max_test.items():
        print("{}:  {}".format(k,v))

def station_train(epoch_num,learningrate,batchsize,device):

    # load data
    inputpath = "pm+gwr_day+time+ev_n.csv"
    df = pd.read_csv(inputpath)
    # 划分10折训练集和测试集
    Station_path = "Stations.csv"
    df_s = pd.read_csv(Station_path)
    train_r2_list = []
    train_rmse_list = []
    test_r2_list = []
    test_rmse_list = []
    test_mae_list = []
    max_train_r2 = 0
    max_train_rmse = 0
    max_train = {}
    max_test_r2 = 0
    max_test_rmse = 0
    max_test_mae = 0
    max_test = {}
    max_epoch_train = 0
    max_epoch_test = 0

    final_train_r2 = 0
    final_train_rmse = 0
    final_test_r2 = 0
    final_test_rmse = 0
    final_test_mae = 0

    final_max_train_r2 = 0
    final_max_train_rmse = 0
    final_max_test_r2 = 0
    final_max_test_rmse = 0
    final_max_test_mae = 0

    n=0
    kf=KFold(n_splits=10,shuffle=True,random_state=1234) #station cv
    for train,test in kf.split(df_s):
        n+=1
        train_station=df_s.iloc[train]
        test_station=df_s.iloc[test]

        train_data = df.loc[df['code'].isin(train_station['code'].tolist())]
        test_data = df.loc[df['code'].isin(test_station['code'].tolist())]
        # load data
        train_dataset=MyDataset_initial(train_data)
        test_dataset=MyDataset_initial(test_data)
        # loader
        train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=0, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=0, drop_last=False)
        # create net
        mynet = MyNet_ini(train_x_size=train_dataset.train_x_len)
        mynet = mynet.to(device)

        # loss function
        loss_fn = nn.MSELoss()
        loss_fn = loss_fn.to(device)

        # optimizer
        lr = learningrate
        optimizer = torch.optim.Adam(mynet.parameters(), lr=lr)
        #train
        epoch = epoch_num
        Epoch_list=[]
        Truevalues=[]
        Estimates=[]
        for i in range(epoch):
            mynet.train()
            for data in train_loader:
                factors, targets= data
                factors=factors.to(device)
                targets = targets.to(device)

                for target in targets.cpu().numpy():
                    Truevalues.append(target)
                outputs = mynet(factors)
                for output in outputs.data.cpu().numpy():
                    Estimates.append(output)

                loss= loss_fn(input= outputs,target=targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #visualization
            Epoch_list.append(i)
            train_r2_list.append(R2(Estimates,Truevalues))
            train_rmse_list.append(rmse(Estimates,Truevalues))
            if train_r2_list[-1] >= max_train_r2:
                max_train_r2= train_r2_list[-1]
                max_train_rmse = train_rmse_list[-1]
                max_epoch_train = i

            if Epoch_list[-1]%10 ==0:
                print("Epoch:{}, train R2:{},RMSE:{}, M".format(Epoch_list[-1],train_r2_list[-1],train_rmse_list[-1]))
            Truevalues.clear()
            Estimates.clear()

            #test
            mynet.eval()
            with torch.no_grad():
                for data in test_loader:
                    factors, targets = data
                    factors = factors.to(device)
                    targets = targets.to(device)

                    for target in targets.cpu().numpy():
                        Truevalues.append(target)

                    outputs=mynet(factors)
                    for output in outputs.cpu().numpy():
                        Estimates.append(output)
                test_r2_list.append(R2(Estimates,Truevalues))
                test_rmse_list.append(rmse(Estimates,Truevalues))
                test_mae_list.append((mae(Estimates,Truevalues)))
                if test_r2_list[-1] >= max_test_r2:
                    max_test_r2 = test_r2_list[-1]
                    max_test_rmse = test_rmse_list[-1]
                    max_epoch_test = i

                if Epoch_list[-1]%10==0:
                    print("epoch: {}, test R2:{},RMSE:{},MAE:{}".format(Epoch_list[-1],test_r2_list[-1], test_rmse_list[-1], test_mae_list[-1]))

                Truevalues.clear()
                Estimates.clear()

        max_train["{}:epoch{}:train_max_r2".format(n, max_epoch_train)] = max_train_r2
        max_train["{}:epoch{}:train_max_rmse".format(n, max_epoch_train)] = max_train_rmse
        max_test["{}:epoch{}:test_max_r2".format(n, max_epoch_test)] = max_test_r2
        max_test["{}:epoch{}:test_max_rmse".format(n, max_epoch_test)] = max_test_rmse
        max_test["{}:epoch{}:test_max_mae".format(n, max_epoch_test)] = max_test_mae

        final_max_train_r2 += max_train_r2
        final_max_train_rmse += max_train_rmse
        final_max_test_r2 += max_test_r2
        final_max_test_rmse += max_test_rmse
        final_max_test_mae += max_test_mae

        max_train_r2 = 0
        max_train_rmse = 0
        max_test_r2 = 0
        max_test_rmse = 0
        max_test_mae = 0

        final_train_r2 = final_train_r2 + train_r2_list[-1]
        final_train_rmse = final_train_rmse + train_rmse_list[-1]
        final_test_r2 = final_test_r2 + test_r2_list[-1]
        final_test_rmse = final_test_rmse + test_rmse_list[-1]
        final_test_mae = final_test_mae + test_mae_list[-1]

    final_train_r2 = final_train_r2 / 10
    final_train_rmse = final_train_rmse / 10
    final_test_r2 = final_test_r2 / 10
    final_test_rmse = final_test_rmse / 10
    final_test_mae = final_test_mae / 10

    final_max_train_r2 = final_max_train_r2 / 10
    final_max_train_rmse = final_max_train_rmse / 10
    final_max_test_r2 = final_max_test_r2 / 10
    final_max_test_rmse = final_max_test_rmse / 10
    final_max_test_mae = final_max_test_mae / 10

    print("10 folds train R2:{},    RMSE:{}".format(final_train_r2, final_train_rmse))
    print("10 folds CV R2:{},    RMSE:{},   MAE: {}".format(final_test_r2, final_test_rmse, final_test_mae))

    print("10 folds max_train R2:{},    RMSE:{}".format(final_max_train_r2, final_max_train_rmse))
    print(
        "10 folds max_CV R2:{},    RMSE:{}, MAE:{}".format(final_max_test_r2, final_max_test_rmse, final_max_test_mae))

    for k, v in max_train.items():
        print("{}:  {}".format(k, v))

    for k, v in max_test.items():
        print("{}:  {}".format(k, v))



#sample_train(1000,0.005,640,device)
station_train(1000,0.005,640,device)