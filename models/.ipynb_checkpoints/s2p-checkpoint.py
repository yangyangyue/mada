import sys
sys.path.append('..')
from dataset import vars

import torch  
import torch.nn as nn  
  
class S2pNet(nn.Module):  
    def __init__(self):  
        super().__init__()  
        self.conv1 = nn.Conv1d(1, 30, kernel_size=11, stride=1, padding=5)  
        self.conv2 = nn.Conv1d(30, 30, kernel_size=9, stride=1, padding=4)  
        self.conv3 = nn.Conv1d(30, 40, kernel_size=7, stride=1, padding=3)  
        self.conv4 = nn.Conv1d(40, 50, kernel_size=5, stride=1, padding=2)  
        self.conv5 = nn.Conv1d(50, 50, kernel_size=5, stride=1, padding=2)  
        self.flatten = nn.Flatten()  
        self.fc1 = nn.Linear(50 * vars.WINDOW_SIZE, 1024)
        self.fc2 = nn.Linear(1024, 1)  
  
    def forward(self, ids, x, tags=None, y_hat=None, weights=None):  
        x = self.conv1(x[:, None, :]).relu()
        x = self.conv2(x).relu()
        x = self.conv3(x).relu()
        x = self.conv4(x).relu()
        x = self.conv5(x).relu()
        x = self.flatten(x)  
        x = self.fc1(x).relu()
        y = self.fc2(x)
        mid = vars.WINDOW_SIZE//2
        return ((y-y_hat[:, mid:mid+1]) ** 2).mean() if self.training else y
