import torch
import warnings
import torch.nn as nn
import numpy as np

from RBM import *
from torch.utils.data import  TensorDataset,DataLoader,Dataset
from torch.optim import Adam, SGD
from Metric import *
from Ori_Resnet_dataset import *

class DBN(nn.Module):
    def __init__(self,hidden_units,visiable_units=256, output_units=1,k=2,
                 learning_rate = 0.1, learning_rate_decay=False,
                 increase_to_cd_k=False, device='cpu'):
        super(DBN,self).__init__()

        self.n_layers= len(hidden_units)
        self.rbm_layers=[]
        self.rbm_nodes=[]
        self.device = device
        self.is_pretrained = False
        self.is_finetune = False

        # Creating different RBM layers
        for i in range(self.n_layers):
            if i ==0:
                input_size =  visiable_units
            else:
                input_size = hidden_units[i-1]
            rbm = RBM(visible_units= input_size, hidden_units= hidden_units[i],
                      k=k,learning_rate=learning_rate,
                      learning_rate_decay=learning_rate_decay,
                      increase_to_cd_k=increase_to_cd_k,device= device)

            self.rbm_layers.append(rbm)

        self.W_rec = [self.rbm_layers[i].weight for i in range(self.n_layers)]
        self.bias_rec = [self.rbm_layers[i].h_bias for i in range(self.n_layers)]

        for i in range(self.n_layers):
            self.register_parameter('W_rec%i'%i, self.W_rec[i])
            # 向我们建立的网络module添加 parameter
            # 将一个不可训练的类型Tensor转换成可以训练的类型parameter，并将这个parameter绑定到这个module里面，
            # 相当于变成了模型的一部分，成为了模型中可以根据训练进行变化的参数。
            self.register_parameter('bias_rec%i'%i, self.bias_rec[i])

        self.bpnn = torch.nn.Linear(hidden_units[-1],output_units).to(self.device)

    def forward(self,input_data):
        """
         running a single forward process.

         Args:
             input_data: Input data of the first RBM layer. Shape:
                 [batch_size, input_length]

         Returns: Output of the last RBM hidden layer.

         """
        v = input_data.to(self.device)
        hid_output = v.clone()
        for i in range(len(self.rbm_layers)):
            hid_output, _ = self.rbm_layers[i].to_hidden((hid_output))
        output = self.bpnn(hid_output)
        return output

    def reconstruct(self, input_data):
        """
        Go forward to the last layer and then go feed backward back to the
        first layer.

        Args:
            input_data: Input data of the first RBM layer. Shape:
                [batch_size, input_length]

        Returns: Reconstructed output of the first RBM visible layer.

        """
        h = input_data.to(self.device)
        p_h = 0
        for i in range(len(self.rbm_layers)):
            p_h,h = self.rbm_layers[i].to_hidden(h)

        for i in range(len(self.rbm_layers)-1,-1,-1):
            p_h,h = self.rbm_layers[i].to_visible(h)

        return p_h,h

    def pretrain(self,x,epoch=50,batch_size=10):
        """
        Train the DBN model layer by layer and fine-tuning with regression
        layer.

        Args:
            x: DBN model input data. Shape: [batch_size, input_length]
            epoch: Train epoch for each RBM.
            batch_size: DBN train batch size.

        Returns:

        """
        hid_output_i = torch.as_tensor(x,dtype=torch.double, device=self.device)
        #data=MyDataset()
        for i in range(len(self.rbm_layers)):
            print("Training rbm layer {}.".format(i+1))

            dataset_i = TensorDataset(hid_output_i)
            data_loader_i = DataLoader(dataset_i,batch_size=batch_size, drop_last=False)

            self.rbm_layers[i].train_rbm(data_loader_i,epoch)
            hid_output_i, _ = self.rbm_layers[i].forward(hid_output_i)

        # Set pretrain finish flag.
        self.is_pretrained = True
        return

    def pretrain_single(self,x, layer_loc, epoch, batch_size):
        """
         Train the ith layer of DBN model.

         Args:
             x: Input of the DBN model.
             layer_loc: Train layer location.
             epoch: Train epoch.
             batch_size: Train batch size.

         Returns:

         """
        if layer_loc>len(self.rbm_layers) or layer_loc <=0:
            raise ValueError('Layer index out of range. ')
        ith_layer = layer_loc-1
        hid_output_i = torch.tensor(x,dtype= torch.double, device=self.device)

        for ith in range(ith_layer):
            hid_output_i,_ = self.rbm_layers[ith].forward(hid_output_i)

        dataset_i = TensorDataset(hid_output_i)
        dataloader_i = DataLoader(dataset_i,batch_size=batch_size,drop_last=False)

        self.rbm_layers[ith_layer].train_rbm(dataloader_i,epoch)
        hid_output_i, _ = self.rbm_layers(ith_layer).forward(hid_output_i)
        return

    def finetune(self,df,test_data,epoch,batch_size,loss_function,optimizer,shuffle=True):
        """
        Fine-tune the train dataset.

        Args:
            df: Input data
            epoch: Fine-tuning epoch
            batch_size: Train batch size
            loss_function: Train loss function
            optimizer: Finetune optimizer
            shuffle: True if shuffle train data

        Returns:

        """
        estimates=[]
        truevalues=[]
        r2_list=[]
        rmse_list=[]
        if not self.is_pretrained:
            warnings.warn("Hasn't pretrained DBN model yet. Recommend "
                          "run self.pretrain() first.", RuntimeWarning)

        dataset = FineTuningDataset(df)
        dataloader = DataLoader(dataset, batch_size,shuffle=shuffle)

        print('Begin fine-tuning. ')
        for epoch_i in range(1,epoch+1):
            total_loss = 0
            for data in dataloader:
                factors, targets = data
                for target in targets.numpy():
                    truevalues.append(target)

                factors = factors.to(self.device)
                targets = targets.to(self.device)

                output = self.forward(factors).squeeze(1)
                for ot in output.data.cpu().numpy():
                    estimates.append(ot)

                loss = loss_function(targets,output)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            r2 = R2(estimates, truevalues)
            r2_list.append(r2)
            rmses = rmse(estimates, truevalues)
            rmse_list.append(rmses)
            if epoch_i%20 == 0:
                print('Epoch:{0}/{1} -rbm_train_R2:{2}, rbm_train_RMSE:{3}' .format(epoch_i, epoch, r2 ,rmses))
                self.test(test_data, batch_size)
            estimates.clear()
            truevalues.clear()

        self.is_finetune = True


        return r2_list[-1],rmse_list[-1]

    def test(self,test_x,batch_size,shuffle=False):
        """
        Predict

        Args:
            test_x: DBN input data. Type: dataframe. Shape: (batch_size, visible_units)
            batch_size: Batch size for DBN model.
            shuffle: True if shuffle predict input data.

        Returns: Prediction result. Type: torch.tensor(). Device is 'cpu' so
            it can transferred to ndarray.
            Shape: (batch_size, output_units)

        """
        if not self.is_pretrained:
            warnings.warn("Hasn't pretrained DBN model yet. Recommend "
                          "run self.pretrain() first.", RuntimeWarning)

        if not self.is_pretrained:
            warnings.warn("Hasn't finetuned DBN model yet. Recommend "
                          "run self.finetune() first.", RuntimeWarning)
        y_predict = torch.tensor([])
        dataset = MyDataset(test_x)
        estimates=[]
        truevalues=[]
        # x_tensor = torch.tensor(test_x,dtype=torch.float, device=self.device)
        # dataset = TensorDataset(x_tensor)
        dataloader = DataLoader(dataset, batch_size,shuffle)
        with torch.no_grad():
            for data in dataloader:
                factors, targets = data
                for target in targets.numpy():
                    truevalues.append(target)

                factors = factors.to(self.device)
                targets = targets.to(self.device)

                output=self.forward(factors)
                for ot in output.data.cpu().numpy():
                    estimates.append(ot)

                #y_predict = torch.cat((y_predict,y.cpu()),0)
            r2 = R2(estimates, truevalues)
            rmses = rmse(estimates, truevalues)
            print('-rbm_test_R2:{}, rbm_test_RMSE:{}'.format( r2, rmses))
            estimates.clear()
            truevalues.clear()
        return r2, rmses


class FineTuningDataset(Dataset):
    """
    Dataset class for whole dataset. x: input data. y: output data
    """
    def __init__(self,df):
        super().__init__()
        self.x = torch.from_numpy(np.array(df.loc[:,["AOD","NDVI","RH","TS","PS","PBLH","ROAD","FACT","DEM"]]))
        self.y = torch.from_numpy(np.array(df.loc[:,"PM"]))
        self.len = len(df)

    def __getitem__(self, index):
        return self.x[index],self.y[index]

    def __len__(self):
        return self.len






