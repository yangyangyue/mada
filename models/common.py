"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

import torch
from torch import nn

class IbnNet(nn.Module):
    def __init__(self, in_channels, out_channels, use_ins=False):
        super().__init__()
        self.use_ins = use_ins
        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = out_channels // 4
        self.stream = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.Conv1d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels)
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
    
class ExampleEncoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.lstm = nn.LSTM(1, out_channels // 2, batch_first=True, bidirectional=True)

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        return output[:, -1, :]
    
class Attention(nn.Module):
    def __init__(self, channels=None) -> None:
        super().__init__()
        self.channels = channels
        if self.channels:
            self.qc = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)
            self.kc = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)
            self.vc = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, q, k, v):
        d_model = q.shape[1]
        if self.channels:
            q = self.qc(q)
            k = self.kc(k)
            v = self.vc(v)
        atten = torch.einsum('ndq,ndk->nqk', q, k)
        atten = atten / (d_model ** 0.5)
        atten = torch.softmax(atten, dim=-1)
        return torch.einsum('nqk,ndk->ndq', atten, v)
