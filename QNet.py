import torch
import torch.nn as nn
import torch.nn.functional as F

from use_cuda import *

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

class QNet(nn.Module):
    def __init__(self, num_actions, k):
        super(QNet, self).__init__()  

        self.conv1 = nn.Conv2d(in_channels=k, out_channels=16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)

        self.lin1 = nn.Linear(in_features=1152, out_features=256)
        self.lin2 = nn.Linear(in_features=256, out_features=num_actions)

        if USE_CUDA:
            self.cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x) 

        x = torch.flatten(x, start_dim = 1)
        
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return x