import torch
from matplotlib import pyplot as plt
from Metric import *
from RNN import *
from Data_load import *

#datatype set to double
torch.set_default_tensor_type(torch.DoubleTensor)

#train device
device = torch.device("cuda")

#load data
inputpath="pm.csv"
data = MyDataset(inputpath)
train_x_size=data.train_x_size
#划分训练集和测试集
torch.manual_seed(0)
train_data_size,test_data_size= round(0.8*data.len),round(0.2*data.len)
train_data, test_data = random_split(data,[train_data_size,test_data_size])

# loader
train_loader = DataLoader(train_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)
test_loader = DataLoader(test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)

# create net
myrnn=MyRNN()
myrnn.to(device)

#loss function
loss_fn = nn.MSELoss()
loss_fn=loss_fn.to(device)

#optimizer
lr =0.01
optimizer = torch.optim.Adam(myrnn.parameters(),lr=lr)

#train
h_state=None
epoch = 200
total_train_step=0
epoch_list=[]
loss_list=[]
y_e_list=[]
y_t_list=[]
R2_list=[]
RMSE_list=[]
test_R2_list=[]
test_RMSE_list=[]
for i in range(epoch):
    print("--第{} train start! ".format(i + 1))
    myrnn.train()
    for data in train_loader:
        factors, targets = data #[4,9]
        factors = factors.view(len(factors), -1, 1)
        #print(factors.shape) #[4,9,1]
        factors = factors.to(device)

        for target in targets.numpy():
            y_t_list.append(target)

        targets = targets.to(device)
        predictions,h_state = myrnn(factors,h_state)
        #print(h_state.shape)#[1,4,32]
        #print(predictions.shape)#[4,9,1]
        h_state=h_state.detach()#将每一次输出的中间状态传递下去(不带梯度)RNN的隐藏状态不参与到模型运算中

        # estimates
        estimates = predictions.data.cpu().numpy()

        for estimate in estimates:
            y_e_list.append(estimate)

        loss= loss_fn(predictions,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("train:{},loss={}".format(total_train_step, loss.item()))

    # R2,RMSE
    epoch_list.append(i)
    R2_list.append(R2(y_e_list, y_t_list))
    RMSE_list.append(rmse(y_e_list, y_t_list))
    y_e_list.clear()
    y_t_list.clear()

    # test
    myrnn.eval()
    with torch.no_grad():
        for data in test_loader:
            factors, targets = data
            factors = factors.view(len(factors), -1, 1)
            factors = factors.to(device)

            for target in targets.numpy():
                y_t_list.append(target)

            targets = targets.to(device)

            predictions,h_state = myrnn(factors,h_state)

            estimates = predictions.data.cpu().numpy()

            for estimate in estimates:
                y_e_list.append(estimate)

            h_state = h_state.detach()  # 将每一次输出的中间状态传递下去(不带梯度)
            loss = loss_fn(input=predictions, target=targets)

        test_R2_list.append(R2(y_e_list, y_t_list))
        test_RMSE_list.append(rmse(y_e_list, y_t_list))
        y_e_list.clear()
        y_t_list.clear()

        print("test R2:{},RMSE:{}".format(test_R2_list[-1], test_RMSE_list[-1]))

# draw img
print(RMSE_list[-1],R2_list[-1])
plt.plot(epoch_list, R2_list,epoch_list,RMSE_list)
plt.xlabel("epoch")
plt.ylabel("Y")
plt.legend(['R2','RMSE'])
plt.show()
