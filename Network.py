import numpy as np
import pandas as pd
import torch
import torch.nn as nn
#import matplotlib.pyplot as plt
#import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.utils.data import DataLoader, TensorDataset, random_split
#from torch.utils.tensorboard import SummaryWriter
#import time

class mynetwork(nn.Module):
    def __init__(self, middle_width, drop_out):
        super().__init__()
        self.linear = nn.Linear(11, int(middle_width/2))
        self.m = nn.Sigmoid()
        self.r = nn.ReLU()
        self.s = nn.Softmax(dim=1)
        self.linear1 = nn.Linear(int(middle_width/2), middle_width)
        self.linear2 = nn.Linear(middle_width, int(middle_width/2))
        self.linear3 = nn.Linear(int(middle_width/2), 1)
        self.drop = nn.Dropout(drop_out)

    def forward(self, xb):
        xb = self.linear(xb)
        xb = self.r(xb)
        xb = self.linear1(xb)
        xb = self.drop(xb)
        xb = self.r(xb)
        xb = self.linear2(xb)
        xb = self.r(xb)
        out = self.linear3(xb)
        #out = self.s(xb)
        return out

def getdata(filename):
    mydataNump = np.genfromtxt(filename, delimiter=';', skip_header=2)
    winedata = mydataNump[:-100, 0:11]
    wineData_tensor = torch.from_numpy(winedata).float()
    quality = mydataNump[:-100, 11]
    quality_tensor = torch.from_numpy(quality).unsqueeze(1).float()
    train_tensor = data_utils.TensorDataset(wineData_tensor, quality_tensor)
    return train_tensor

def getevaldata(filename):
    mydataNump = np.genfromtxt(filename, delimiter=';', skip_header=2)
    winedata = mydataNump[-100:, 0:11]
    wineData_tensor = torch.from_numpy(winedata).float()
    quality = mydataNump[-100:, 11]
    quality_tensor = torch.from_numpy(quality).unsqueeze(1).float()
    train_tensor = data_utils.TensorDataset(wineData_tensor, quality_tensor)
    return train_tensor