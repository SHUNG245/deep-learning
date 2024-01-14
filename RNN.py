import torch
import numpy as np
from torch import nn


class MyRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn=nn.RNN(
            input_size= 1,
            hidden_size= 32,
            num_layers=2,
            batch_first=True
        )
        self.fc=nn.Linear(32,1)

    def forward(self,x,h_state):
        r_out,h_state=self.rnn(x,h_state)
        r_out=r_out[:,-1,:]
        predition = self.fc(r_out)
        predition = predition.squeeze(-1)
        return predition.squeeze(-1),h_state

