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
path = "NN/"
csv_path = 'NN/FSTDNN_csv'
#train device
device = torch.device("cuda")

def sample_train(epoch_num,learningrate,batchsize,ev_num,day_num,month_num,year_num,SW,device):
    #load data
    inputpath="pm+gwr_day+time+ev.csv"
    df=pd.read_csv(inputpath)
    #划分10折训练集和测试集

    estimates=[]
    observations =[]
    dems_total=[]
    evs_total=[]
    year_total=[]
    month_total =[]
    day_total=[]

    train_r2_list=[]
    train_rmse_list=[]
    train_mae_list=[]
    test_r2_list=[]
    test_rmse_list=[]
    test_mae_list=[]

    max_train_r2=0
    max_train_rmse=0
    max_train_mae= 0
    max_train={}
    max_test_r2=0
    max_test_rmse = 0
    max_test_mae = 0
    max_test={}
    max_epoch_train=0
    max_epoch_test=0

    final_train_r2=0
    final_train_rmse=0
    final_train_mae = 0
    final_test_r2=0
    final_test_rmse=0
    final_test_mae = 0

    final_max_train_r2=0
    final_max_train_rmse=0
    final_max_train_mae=0
    final_max_test_r2=0
    final_max_test_rmse=0
    final_max_test_mae=0

    n=0
    kf=KFold(n_splits=10,shuffle=True,random_state=345) #sample cv
    for train,test in kf.split(df):
        n+=1
        train_data=df.iloc[train]
        test_data=df.iloc[test]
        #load data
        train_dataset =STDNN_Dataset(train_data)
        test_dataset = STDNN_Dataset(test_data)
        #loader
        train_loader = DataLoader(train_dataset,batch_size= batchsize,shuffle=True,num_workers=0,drop_last=False)
        test_loader = DataLoader(test_dataset,batch_size= test_data.shape[0], shuffle= True, num_workers= 0, drop_last= False)
        #create net
        STDNN = STDNnet(train_dataset.train_x_len,ev_num=ev_num,day=day_num,month=month_num,year=year_num)
        STDNN=STDNN.to(device)

        #loss function
        loss_fn = StationweightedLoss()
        loss_fn = loss_fn.to(device)

        #optimizer
        params = [p for p in STDNN.parameters() if p.requires_grad]
        print(STDNN.parameters())
        lr= learningrate
        optimizer=torch.optim.Adam(params,lr= lr)
        #train
        epoch = epoch_num
        Epoch_list=[]
        Truevalues=[]
        Estimates=[]
        DEMS=[]
        EVS=[]
        YEARS=[]
        MONTHS=[]

        for i in range(epoch+1):
            STDNN.train()
            for data in train_loader:
                factors, targets, evs, years, months, days,weight_dem,year_c,month_c= data
                factors = factors.to(device)
                targets = targets.to(device)
                evs = evs.to(device)
                years = years.to(device)
                months = months.to(device)
                days = days.to(device)
                weight_dem = weight_dem.to(device)
                #weight_station = weight_station.to(device)

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

            if Epoch_list[-1] % 10 == 0:
                print("Epoch:{}, train R2:{},RMSE:{}, MAE:{}".format(Epoch_list[-1], train_r2_list[-1], train_rmse_list[-1],train_mae_list[-1]))
            Truevalues.clear()
            Estimates.clear()

            #test
            STDNN.eval()
            with torch.no_grad():
                for data in test_loader:
                    factors, targets, evs, years, months, days,weight_dem,year_c,month_c = data
                    factors = factors.to(device)
                    targets = targets.to(device)
                    evs = evs.to(device)
                    years = years.to(device)
                    months = months.to(device)
                    days = days.to(device)

                    for target in targets.cpu().numpy():
                        Truevalues.append(target)
                        # output
                    outputs = STDNN(factors, evs, days, months, years)
                    for output in outputs.data.cpu().numpy():
                        Estimates.append(output)
                    for ev in evs.data.cpu().numpy():
                        EVS.append(ev[0])
                    for dem in factors.data.cpu().numpy():
                        DEMS.append(dem[8])
                    for year in year_c.data.numpy():
                        YEARS.append(year)
                    for month in month_c.data.numpy():
                        MONTHS.append(month)

                test_r2_list.append(R2(Estimates, Truevalues))
                test_rmse_list.append(rmse(Estimates, Truevalues))
                test_mae_list.append(mae(Estimates,Truevalues))

                if test_r2_list[-1] >= max_test_r2:
                    max_test_r2 = test_r2_list[-1]
                    max_test_rmse = test_rmse_list[-1]
                    max_test_mae = test_mae_list[-1]
                    max_epoch_test = i
                if i == 950 :
                    #torch.save(STDNN,path+"ISTDNN"+n+"_sample.pt")
                    for e,o,d,ev,y,m in zip(Estimates,Truevalues,DEMS,EVS,YEARS,MONTHS):
                        estimates.append(e)
                        observations.append(o)
                        dems_total.append(d)
                        evs_total.append(ev)
                        year_total.append(y)
                        month_total.append(m)
                if Epoch_list[-1] % 10 == 0:
                    print("epoch: {}, test R2:{},RMSE:{}, MAE:{}".format(Epoch_list[-1], test_r2_list[-1], test_rmse_list[-1],test_mae_list[-1]))
                    #print(Estimates[0:5])
                    #print(Truevalues[0:5])
                Truevalues.clear()
                Estimates.clear()
                DEMS.clear()
                EVS.clear()
                YEARS.clear()
                MONTHS.clear()

        max_train["{}:epoch{}:train_max_r2".format(n, max_epoch_train)] = max_train_r2
        max_train["{}:epoch{}:train_max_rmse".format(n, max_epoch_train)] = max_train_rmse
        max_train["{}:epoch{}:train_max_mae".format(n, max_epoch_train)] = max_train_mae
        max_test["{}:epoch{}:test_max_r2".format(n, max_epoch_test)] = max_test_r2
        max_test["{}:epoch{}:test_max_rmse".format(n, max_epoch_test)] = max_test_rmse
        max_test["{}:epoch{}:test_max_mae".format(n, max_epoch_test)] = max_test_mae

        final_max_train_r2 += max_train_r2
        final_max_train_rmse += max_train_rmse
        final_max_train_mae += max_train_mae
        final_max_test_r2 += max_test_r2
        final_max_test_rmse += max_test_rmse
        final_max_test_mae += max_test_mae

        max_train_r2 = 0
        max_train_rmse = 0
        max_train_mae = 0
        max_test_r2 = 0
        max_test_rmse = 0
        max_test_mae = 0

        final_train_r2 = final_train_r2 + train_r2_list[-1]
        final_train_rmse = final_train_rmse + train_rmse_list[-1]
        final_train_mae = final_train_mae + train_mae_list[-1]
        final_test_r2 = final_test_r2 + test_r2_list[-1]
        final_test_rmse = final_test_rmse + test_rmse_list[-1]
        final_test_mae = final_test_mae + test_mae_list[-1]

    final_train_r2 = final_train_r2 / 10
    final_train_rmse = final_train_rmse / 10
    final_train_mae = final_train_mae / 10
    final_test_r2 = final_test_r2 / 10
    final_test_rmse = final_test_rmse / 10
    final_test_mae = final_test_mae / 10

    final_max_train_r2 = final_max_train_r2 / 10
    final_max_train_rmse = final_max_train_rmse / 10
    final_max_train_mae = final_max_train_mae / 10
    final_max_test_r2 = final_max_test_r2 / 10
    final_max_test_rmse = final_max_test_rmse / 10
    final_max_test_mae = final_max_test_mae / 10

    print("10 folds train R2:{},    RMSE:{},    MAE:{}".format(final_train_r2, final_train_rmse, final_train_mae))
    print("10 folds CV R2:{},    RMSE:{},     MAE:{}".format(final_test_r2, final_test_rmse, final_test_mae))

    print("10 folds max_train R2:{},    RMSE:{},     MAE:{}".format(final_max_train_r2, final_max_train_rmse, final_max_train_mae))
    print("10 folds max_CV R2:{},    RMSE:{},     MAE:{}".format(final_max_test_r2, final_max_test_rmse, final_max_test_mae))

    for k, v in max_train.items():
        print("{}:  {}".format(k, v))

    for k, v in max_test.items():
        print("{}:  {}".format(k, v))
    df_e = pd.DataFrame(estimates)
    df_o = pd.DataFrame(observations)
    df_dem = pd.DataFrame(dems_total)
    df_ev = pd.DataFrame(evs_total)
    df_year = pd.DataFrame(year_total)
    df_month = pd.DataFrame(month_total)
    df = pd.concat([df_e,df_o,df_dem,df_ev,df_year,df_month],axis=1)
    df.to_csv(csv_path+'/sample_year.csv',index=False)
def station_train(epoch_num,learningrate,batchsize,ev_num,day_num,month_num,year_num,SW,device):
    # load data
    inputpath = "pm+gwr_day+time+ev.csv"
    df = pd.read_csv(inputpath)
    # 划分10折训练集和测试集
    Station_path = "Stations.csv"
    df_s = pd.read_csv(Station_path)

    estimates=[]
    observations=[]
    dems_total=[]
    evs_total=[]
    year_total=[]
    month_total =[]
    day_total=[]

    train_r2_list = []
    train_rmse_list = []
    train_mae_list = []
    test_r2_list = []
    test_rmse_list = []
    test_mae_list = []

    max_train_r2 = 0
    max_train_rmse = 0
    max_train_mae = 0
    max_train = {}
    max_test_r2 = 0
    max_test_rmse = 0
    max_test_mae = 0
    max_test = {}
    max_epoch_train = 0
    max_epoch_test = 0

    final_train_r2 = 0
    final_train_rmse = 0
    final_train_mae = 0
    final_test_r2 = 0
    final_test_rmse = 0
    final_test_mae = 0

    final_max_train_r2 = 0
    final_max_train_rmse = 0
    final_max_train_mae = 0
    final_max_test_r2 = 0
    final_max_test_rmse = 0
    final_max_test_mae = 0

    n = 0
    kf = KFold(n_splits=10, shuffle=True, random_state=345)  # sample cv
    for train, test in kf.split(df_s):
        n += 1
        train_station=df_s.iloc[train]
        test_station=df_s.iloc[test]
        print(test_station['code'])
        train_data = df.loc[df['code'].isin(train_station['code'].tolist())]
        test_data = df.loc[df['code'].isin(test_station['code'].tolist())]
        # load data
        train_dataset = STDNN_Dataset(train_data)
        test_dataset = STDNN_Dataset(test_data)
        # loader
        train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=0, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=0, drop_last=False)
        # create net
        STDNN = STDNnet(train_dataset.train_x_len, ev_num=ev_num, day=day_num, month=month_num, year=year_num)
        STDNN = STDNN.to(device)

        # loss function
        loss_fn = StationweightedLoss()
        #loss_fn = nn.MSELoss()
        loss_fn = loss_fn.to(device)

        # optimizer
        params = [p for p in STDNN.parameters() if p.requires_grad]
        print(STDNN.parameters())
        lr = learningrate
        optimizer = torch.optim.Adam(params, lr=lr)
        # train
        epoch = epoch_num
        Epoch_list = []
        Truevalues = []
        Estimates = []
        DEMS=[]
        EVS=[]
        YEARS=[]
        MONTHS=[]

        for i in range(epoch + 1):
            STDNN.train()
            for data in train_loader:
                factors, targets, evs, years, months, days,weight_dem,year_c,month_c = data
                factors = factors.to(device)
                targets = targets.to(device)
                evs = evs.to(device)
                years = years.to(device)
                months = months.to(device)
                days = days.to(device)
                weight_dem = weight_dem.to(device)
                #weight_station = weight_station.to(device)

                for target in targets.cpu().numpy():
                    Truevalues.append(target)
                # output
                outputs = STDNN(factors, evs, days, months, years)

                for output in outputs.data.cpu().numpy():
                    Estimates.append(output)

                # loss compute
                loss = loss_fn(outputs,targets,weight_dem,super_weight=SW)
                #loss = loss_fn(outputs,targets)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # visualization
            Epoch_list.append(i)
            train_r2_list.append(R2(Estimates, Truevalues))
            train_rmse_list.append(rmse(Estimates, Truevalues))
            train_mae_list.append(mae(Estimates, Truevalues))
            if train_r2_list[-1] >= max_train_r2:
                max_train_r2 = train_r2_list[-1]
                max_train_rmse = train_rmse_list[-1]
                max_train_mae = train_mae_list[-1]
                max_epoch_train = i


            if Epoch_list[-1] % 10 == 0:
                print("Epoch:{}, train R2:{},RMSE:{}, MAE:{}".format(Epoch_list[-1], train_r2_list[-1],
                                                                     train_rmse_list[-1], train_mae_list[-1]))
            Truevalues.clear()
            Estimates.clear()

            # test
            STDNN.eval()
            with torch.no_grad():
                for data in test_loader:
                    factors, targets, evs, years, months, days,weight_dem,year_c,month_c = data
                    factors = factors.to(device)
                    targets = targets.to(device)
                    evs = evs.to(device)
                    years = years.to(device)
                    months = months.to(device)
                    days = days.to(device)

                    for target in targets.cpu().numpy():
                        Truevalues.append(target)
                        # output
                    outputs = STDNN(factors, evs, days, months, years)
                    for output in outputs.data.cpu().numpy():
                        Estimates.append(output)

                    for ev in evs.data.cpu().numpy():
                        EVS.append(ev[0])
                    for dem in factors.data.cpu().numpy():
                        DEMS.append(dem[8])
                    for year in year_c.data.numpy():
                        YEARS.append(year)
                    for month in month_c.data.numpy():
                        MONTHS.append(month)

                test_r2_list.append(R2(Estimates, Truevalues))
                test_rmse_list.append(rmse(Estimates, Truevalues))
                test_mae_list.append(mae(Estimates, Truevalues))

                if test_r2_list[-1] >= max_test_r2:
                    max_test_r2 = test_r2_list[-1]
                    max_test_rmse = test_rmse_list[-1]
                    max_test_mae = test_mae_list[-1]
                    max_epoch_test = i
                if i == 950:
                    for e,o,d,ev,y,m in zip(Estimates,Truevalues,DEMS,EVS,YEARS,MONTHS):
                        estimates.append(e)
                        observations.append(o)
                        dems_total.append(d)
                        evs_total.append(ev)
                        year_total.append(y)
                        month_total.append(m)

                if Epoch_list[-1] % 10 == 0:
                    print("epoch: {}, test R2:{},RMSE:{}，  MAE：{}".format(Epoch_list[-1], test_r2_list[-1], test_rmse_list[-1],test_mae_list[-1]))
                    #print(Estimates[0:5])
                    #print(Truevalues[0:5])
                Truevalues.clear()
                Estimates.clear()
                DEMS.clear()
                EVS.clear()
                YEARS.clear()
                MONTHS.clear()

        max_train["{}:epoch{}:train_max_r2".format(n, max_epoch_train)] = max_train_r2
        max_train["{}:epoch{}:train_max_rmse".format(n, max_epoch_train)] = max_train_rmse
        max_train["{}:epoch{}:train_max_mae".format(n, max_epoch_train)] = max_train_mae
        max_test["{}:epoch{}:test_max_r2".format(n, max_epoch_test)] = max_test_r2
        max_test["{}:epoch{}:test_max_rmse".format(n, max_epoch_test)] = max_test_rmse
        max_test["{}:epoch{}:test_max_mae".format(n, max_epoch_test)] = max_test_mae

        final_max_train_r2 += max_train_r2
        final_max_train_rmse += max_train_rmse
        final_max_train_mae += max_train_mae
        final_max_test_r2 += max_test_r2
        final_max_test_rmse += max_test_rmse
        final_max_test_mae += max_test_mae

        max_train_r2 = 0
        max_train_rmse = 0
        max_train_mae = 0
        max_test_r2 = 0
        max_test_rmse = 0
        max_test_mae = 0

        final_train_r2 = final_train_r2 + train_r2_list[-1]
        final_train_rmse = final_train_rmse + train_rmse_list[-1]
        final_train_mae = final_train_mae + train_mae_list[-1]
        final_test_r2 = final_test_r2 + test_r2_list[-1]
        final_test_rmse = final_test_rmse + test_rmse_list[-1]
        final_test_mae = final_test_mae + test_mae_list[-1]

    final_train_r2 = final_train_r2 / 10
    final_train_rmse = final_train_rmse / 10
    final_train_mae = final_train_mae / 10
    final_test_r2 = final_test_r2 / 10
    final_test_rmse = final_test_rmse / 10
    final_test_mae = final_test_mae / 10

    final_max_train_r2 = final_max_train_r2 / 10
    final_max_train_rmse = final_max_train_rmse / 10
    final_max_train_mae = final_max_train_mae / 10
    final_max_test_r2 = final_max_test_r2 / 10
    final_max_test_rmse = final_max_test_rmse / 10
    final_max_test_mae = final_max_test_mae / 10

    print("10 folds train R2:{},    RMSE:{},    MAE:{}".format(final_train_r2, final_train_rmse, final_train_mae))
    print("10 folds CV R2:{},    RMSE:{},     MAE:{}".format(final_test_r2, final_test_rmse, final_test_mae))

    print("10 folds max_train R2:{},    RMSE:{},     MAE:{}".format(final_max_train_r2, final_max_train_rmse,
                                                                    final_max_train_mae))
    print("10 folds max_CV R2:{},    RMSE:{},     MAE:{}".format(final_max_test_r2, final_max_test_rmse,
                                                                 final_max_test_mae))
    df_e = pd.DataFrame(estimates)
    df_o = pd.DataFrame(observations)
    df_dem = pd.DataFrame(dems_total)
    df_ev = pd.DataFrame(evs_total)
    df_year = pd.DataFrame(year_total)
    df_month = pd.DataFrame(month_total)
    df = pd.concat([df_e,df_o,df_dem,df_ev,df_year,df_month],axis=1)
    df.to_csv(csv_path+'/station_year_1_sw.csv',index=False)

if __name__ == '__main__':
    #sample_train(1000,0.005,640,5,15,6,2,2,device)
    station_train(1000,0.005,640,5,15,6,2,2,device)


