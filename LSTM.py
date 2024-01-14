import torch
from torch import nn


class MyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm=nn.LSTM(
            input_size= 1,
            hidden_size= 32,
            batch_first=True,
            num_layers=2
        )
        self.linear=nn.Linear(32,1)

    def forward(self,x):
        r_out,(h_n,c_n)=self.lstm(x)
        #r_out [batch,seq_len,hiddensize]
        #h_n [num_layer,batch,hidden_size]
        #c_n [num_layer,batch,hidden_size]
        x#=r_out[:,-1,:]
        x=h_n[-1,:,:]
        x=self.linear(x)
        return x.squeeze(-1)