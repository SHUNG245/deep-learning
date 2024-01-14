import torch
from matplotlib import pyplot as plt
from Metric import *
from Data_load import *
from LSTM import *

#datatype set to double
torch.set_default_tensor_type(torch.DoubleTensor)

#train device
device = torch.device("cuda")

#load data
inputpath="pm_n.csv"
data = MyDataset(inputpath)
train_x_size=data.train_x_size
#划分训练集和测试集
torch.manual_seed(0)
train_data_size,test_data_size= round(0.8*data.len),round(0.2*data.len)
train_data,test_data = random_split(data,[train_data_size,test_data_size])

#dataloader
train_loader=DataLoader(train_data,batch_size=128,num_workers=0,shuffle=True,drop_last=False)
test_loader = DataLoader(test_data,batch_size=128,num_workers=0,shuffle=True,drop_last=False)

#create net
mylstm=MyLSTM()
mylstm.to(device)

#loss_fn
loss_fn=nn.MSELoss()
loss_fn.to(device)

#optimizer
lr=0.01
optimizer= torch.optim.Adam(params=mylstm.parameters(),lr=lr)

#train
epoch = 400
total_train_step=0
epoch_list=[]
y_e_list=[]
y_t_list=[]
R2_list=[]
RMSE_list=[]
test_R2_list=[]
test_RMSE_list=[]
for i in range(epoch):
    print("--第{} train start! ".format(i + 1))

    mylstm.train()
    for data in train_loader:
        factors,targets =data#[4,9],[4,1]
        factors=factors.view(len(factors),-1,1)#[4,9,1]
        factors=factors.to(device)
        #
        for target in targets.numpy():
            y_t_list.append(target)

        targets=targets.to(device)
        predictions=mylstm(factors)
        #estimates
        estimates=predictions.data.cpu().numpy()

        for estimate in estimates:
            y_e_list.append(estimate)

        loss=loss_fn(predictions,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("train:{},loss={}".format(total_train_step, loss.item()))

     #R2,RMSE
    epoch_list.append(i)
    R2_list.append(R2(y_e_list,y_t_list))
    RMSE_list.append(rmse(y_e_list,y_t_list))
    y_e_list.clear()
    y_t_list.clear()

    # test
    mylstm.eval()
    with torch.no_grad():
        for data in test_loader:
            factors, targets = data  # [4,9],[4,1]
            factors = factors.view(len(factors), -1, 1)  # [4,9,1]
            factors = factors.to(device)

            for target in targets.numpy():
                y_t_list.append(target)
            targets = targets.to(device)

            predictions = mylstm(factors)

            # estimates
            estimates = predictions.data.cpu().numpy()
            for estimate in estimates:
                y_e_list.append(estimate)

        test_R2_list.append(R2(y_e_list, y_t_list))
        test_RMSE_list.append(rmse(y_e_list, y_t_list))
        y_e_list.clear()
        y_t_list.clear()
        print("test R2:{},RMSE:{}".format(test_R2_list[-1],test_RMSE_list[-1]))

# draw img
print(RMSE_list[-1],R2_list[-1])
plt.plot(epoch_list, R2_list,epoch_list,RMSE_list)
plt.xlabel("epoch")
plt.ylabel("Y")
plt.legend(['R2','RMSE'])
plt.show()
