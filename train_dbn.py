import torch
import pandas as pd
import numpy as np
from Ori_Resnet_dataset import *
import matplotlib.pyplot as plt
from Metric import *
from sklearn.model_selection import KFold
from DBN import *

#datatype set to double
torch.set_default_tensor_type(torch.DoubleTensor)
#train device
device = torch.device("cuda")
#load data
inputpath="pm_n.csv"
df=pd.read_csv(inputpath)

#设置DBN超参数
input_length = 9
output_length = 1
hidden_units=[256,128,64,32]
batch_size = 640
epoch_pretrain = 50
epoch_finetune = 800
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam
pretrain_lr=0.1
finetuning_lr=0.001
#划分10折训练集和测试集

train_r2_list=[]
train_rmse_list=[]
test_r2_list=[]
test_rmse_list=[]

final_train_r2=0
final_train_rmse=0
final_test_r2=0
final_test_rmse=0
kf=KFold(n_splits=10,shuffle=False)#station cv
#kf=KFold(n_splits=10,shuffle=True,random_state=1234)
for train,test in kf.split(df):
    train_data=df.iloc[train]
    test_data=df.iloc[test]
    #load data
    train_dataset=MyDataset(train_data)
    test_dataset=MyDataset(test_data)
    #build model
    dbn = DBN(hidden_units, input_length, output_length, learning_rate=pretrain_lr, device=device)

    #train model
    dbn.pretrain(train_dataset.x, epoch_pretrain, batch_size=batch_size)
    train_r2, train_rmse = dbn.finetune(train_data,test_data, epoch_finetune, batch_size, loss_function, optimizer(dbn.parameters(),lr=finetuning_lr))
    final_train_r2+=train_r2
    final_train_rmse += train_rmse
    test_r2,test_rmse = dbn.test(test_data,batch_size,shuffle=False)
    final_test_r2+=test_r2
    final_test_rmse+=test_rmse

final_train_r2 = final_train_r2/10
final_train_rmse = final_train_rmse/10
final_test_r2 = final_test_r2/10
final_test_rmse = final_test_rmse/10
print("train_r2: {}, train_rmse: {}.".format(final_train_r2,final_train_rmse))
print("test_r2: {}, test_rmse: {} .".format(final_test_r2,final_test_rmse))

