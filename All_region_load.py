#把整个研究区域的点作为输入读取最终的网络得到结果
import os

import torch
import pandas as pd
import numpy as np
from All_region_loade_dataset import *
import matplotlib.pyplot as plt
from Metric import *
torch.set_default_tensor_type(torch.DoubleTensor)
device='cuda'
# inputpath="/media/suheng/16C2E368EA401D04/" \
#           "Deep_learning/Experimental_data/PM2.5/All-region/ALL_day/day_remove_null_time"
inputpath="/media/suheng/16C2E368EA401D04/" \
           "Deep_learning/Experimental_data/PM2.5/All-region/ALL_day/day_winter"
outputpath="/media/suheng/16C2E368EA401D04/" \
          "Deep_learning/Experimental_data/PM2.5/All-region/ALL_day/day_result_aod"
allpath="/media/suheng/16C2E368EA401D04/" \
          "Deep_learning/Experimental_data/PM2.5/All-region/ALL_day/base"
netPath1="NN/ISTDNN_all/ISTDNN2_sample_aod.pt"
# netPath2="NN/ISTDNN_all/ISTDNN2_sample.pt"
# netPath3="NN/ISTDNN_all/ISTDNN3_sample.pt"
df_all = pd.read_csv(allpath+'/'+'base.csv')
dem_min=0
dem_mm=816
fact_m=0.034478
fact_mm=0.50341
road_m=0.453483
road_mm= 5.178907
ts_m=252.916
ts_mm=54.839
ps_m=880.431*100
ps_mm=163.379*100
rh_m= 9.2953
rh_mm=81.6759
pblh_m=60.6429
pblh_mm=1883.517
ndvi_m=0
ndvi_mm=8840
aod_m=0.015
aod_mm=3.25748
def loadData(i,outfilename,df_all):
    filename='day'+str(i)+'.csv'
    df=pd.read_csv(inputpath+'/'+filename)
    df.rename(columns={'NDVI'+i:'NDVI','PBLH'+i:'PBLH',
               'PS'+i:'PS','RH'+i:'RH',
               'TS'+i:'TS'},inplace=True)
    if df.shape[0]==0: return df_all
    #修改NDVI，PS的值
    df['NDVI']=df['NDVI']/10000
    df['PS'] = df['PS']/100
    df['DEM'] = df['DEM']/2759
    # df['DEM'] = df['DEM']/dem_mm
    # df['FACT'] = (df['FACT']-fact_m)/fact_mm
    # df['ROAD'] = (df['ROAD']-road_m)/road_mm
    # df['TS'] = (df['TS']-ts_m)/ts_mm
    # df['PS'] = (df['PS']-ps_m)/ps_mm
    # df['RH'] = (df['RH']-rh_m)/rh_mm
    # df['PBLH'] = (df['PBLH']-pblh_m)/pblh_mm
    # df['NDVI']=df['NDVI']/ndvi_mm
    # df['AOD'] = (df['AOD']-aod_m)/aod_mm
    #dataload
    dataset=STDNN_Dataset(df)
    dataloader = DataLoader(dataset,batch_size=df.shape[0], shuffle=True, num_workers=0, drop_last=False)

    #model_load
    model1=torch.load(netPath1)
    # model2=torch.load(netPath2)
    # model3=torch.load(netPath3)
    #datasave
    estimates1=[]
    estimates2=[]
    estimates3=[]
    ids=[]
    model1.eval()
    # model2.eval()
    # model3.eval()
    with torch.no_grad():
        for data in dataloader:
            factors, evs, years, months, days,id = data
            factors = factors.to(device)
            evs = evs.to(device)
            years = years.to(device)
            months = months.to(device)
            days = days.to(device)
            # for id in id.numpy():
            #     ids.append(id)
            # output
            outputs1 = model1(factors, evs, days, months, years)
            #outputs2 = model2(factors, evs, days, months, years)
            #outputs3 = model3(factors, evs, days, months, years)
            # for id ,output1,output2,output3 in zip(id.numpy(),outputs1.data.cpu().numpy(),outputs2.data.cpu().numpy(),outputs3.data.cpu().numpy()):
            #     ids.append(id)
            #     estimates1.append(output1)
            #     estimates2.append(output2)
            #     estimates3.append(output3)
            for id,output1 in zip(id.numpy(),outputs1.data.cpu().numpy()):
                ids.append(id)
                estimates1.append(output1)
            # for output2 in outputs2.data.cpu().numpy():
            #     estimates2.append(output2)
        #result = pd.DataFrame({'POINTID':ids,'Net_2':estimates1,'Net_3':estimates2,'Net_4':estimates3})
        result = pd.DataFrame({'POINTID':ids,'Net_2':estimates1})
        #d_r = pd.DataFrame({'POINTID':ids,'day{}'.format(i):estimates1})
        df = pd.merge(df, result, on='POINTID')
        #df_all=pd.merge(df_all,d_r,on='POINTID')
        df.to_csv(outputpath+'/'+outfilename,index=False)
        return df_all


# for i in range(1461):
#     df_all = loadData(i+1,'day'+str(i+1)+'.csv',df_all)
for file in os.listdir(inputpath):
    i =file.split('.')[0].split('day')[1]
    df_all = loadData(i , 'day' + str(i ) + '.csv', df_all)

#df_all.to_csv(allpath+'/'+'day_all.csv',index=False)



