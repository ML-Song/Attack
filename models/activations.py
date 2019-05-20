#coding=utf-8
import torch
from torch import nn


class HardConcrete(nn.Module):
    def __init__(self, loc=0, temp=0.1, gamma=-0.1, zeta=1.1):
        super().__init__()
        self.loc = loc
        self.temp = temp
        self.gamma = gamma
        self.zeta = zeta
        
    def hard_sigmoid(self, x):
        return torch.min(torch.ones_like(x), torch.max(torch.zeros_like(x), x))
    
    def forward(self, x):
        s = torch.sigmoid((torch.log(x) - torch.log(1 - x) + self.loc) / self.temp)
        s = s * (self.zeta - self.gamma) + self.gamma
        return self.hard_sigmoid(s)