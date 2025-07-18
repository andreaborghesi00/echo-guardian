import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(101, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(16, 1) # 2 classes
        self.dropout = nn.Dropout(p=0.5)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(16)

    def forward(self, x):
        x = F.leaky_relu(self.batchnorm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.batchnorm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.batchnorm3(self.fc3(x)))
        x = self.fc4(x)
        return torch.sigmoid(x)
    