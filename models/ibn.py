"""
Reproduced model of VAE

written by lily
email: lily231147@gmail.com
"""
import torch
from torch import nn

class IbnNet(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, use_ins=True):
        super().__init__()
        self.use_ins = use_ins
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stream = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256)
        )
        if self.use_ins:
            self.in_norm = nn.InstanceNorm1d(256)

    def forward(self, x):
        if self.in_channels == self.out_channels:
            x = x + self.stream(x)
        else:
            x = self.stream(x)
        if self.use_ins:
            x = self.in_norm(x)
        return torch.relu(x)