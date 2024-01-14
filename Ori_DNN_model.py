import torch
from torch import nn


class MyNet_ini(nn.Module):
    def __init__(self,train_x_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(train_x_size,128),
            #nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self,x):
        x=self.model(x)
        return x.squeeze(-1)
