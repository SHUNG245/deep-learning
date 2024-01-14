import torch
import Data_load
from torch import nn
import numpy as np

class Ori_Resnet(nn.Module):
    def __init__(self,train_x_size):
        super().__init__()
        self.relu=nn.ReLU()
        self.bn0 = nn.BatchNorm1d(train_x_size)
        self.hd_layer1 = nn.Linear(train_x_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.hd_layer2 = nn.Linear(64,32)
        self.bn2 = nn.BatchNorm1d(32)
        self.hd_layer3 = nn.Linear(32,16)
        self.bn3 = nn.BatchNorm1d(16)
        self.hd_layer4 = nn.Linear(16,32)
        self.bn4 = nn.BatchNorm1d(32)
        self.hd_layer5 = nn.Linear(32,64)
        self.bn5 = nn.BatchNorm1d(64)
        self.hd_layer6 = nn.Linear(64,train_x_size)
        self.bn6 = nn.BatchNorm1d(train_x_size)
        self.hd_layer7 = nn.Linear(train_x_size,1)

    # def tensor_merge(self,x,ev,day,month,year):
    #     x = torch.cat([x, ev,day,month,year], 1)
    #     return x

    def forward(self,x):

        input= self.bn0(x)

        out1 = self.hd_layer1(input)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out2 = self.hd_layer2(out1)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)

        out3 = self.hd_layer3(out2)
        out3 = self.bn3(out3)
        out3 = self.relu(out3)

        out4 = self.hd_layer4(out3)
        out4 = self.bn4(out4)
        out4 = self.relu(out4)

        out5 = self.hd_layer5(out4+out2)
        out5 = self.bn5(out5)
        out5 = self.relu(out5)

        out6 = self.hd_layer6(out5+out1)
        out6 = self.bn6(out6)
        out6 = self.relu(out6)

        output = self.hd_layer7(out6+input)
        return output.squeeze(-1)


