import torch as th
from torch import nn

class Sprout_Dense(nn.Module):
    def __init__(self, hids, outs, inputs):
        super().__init__()
        shape = inputs.shape
        shape_ = inputs.reshape(shape[0],-1).shape
        self.linear1 = nn.Linear(shape[-1], hids)
        self.linear2 = nn.Linear(hids, outs)

    def forward(self, inputs):
        shape = inputs.shape
        inputs = inputs.reshape(shape[0],-1)
        out = self.linear1(inputs)
        out = self.linear2(out)
        return out