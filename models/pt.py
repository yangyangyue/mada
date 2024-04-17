"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

import math
import torch
from torch import nn

from models.common import IbnNet, TransformerEncoder

def pe(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe
    
class UpNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ibn = IbnNet(in_channels, out_channels)
        self.up = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        

    def forward(self, x):
        y = self.ibn(x)
        y = self.up(x)
        return y

class PtNet(nn.Module):
    def __init__(self, channels=256, n_layers=4, window_size=1024, patch_size=8, patch_stride=4) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        n_patch = window_size // patch_stride
        pad_len = window_size - patch_size - (n_patch - 1) * patch_stride
        self.padding = nn.ZeroPad1d((0, pad_len))
        self.tokenlizer = IbnNet(patch_size, channels)
        self.encoder = TransformerEncoder(n_layers)
        self.up_nets = nn.Sequential()
        t_patch_stride = patch_stride
        while t_patch_stride > 1:
            self.up_nets.add(UpNet(channels, channels))
            t_patch_stride >>= 2
        self.detokenlizer = IbnNet(channels, 1)

    def forward(self, _, x, gt_apps=None):
        """
        Args:
            examples (N, 3, L): input examples
            samples (N, L): input samples
        """
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride).permute(0, 2, 1)
        x = self.padding(x) # (N, patch_size, n_patch)
        x = self.tokenlizer(x).permute(0, 2, 1) # (N, n_patch, C)
        x = x + pe(x.shape[1], x.shape[2]).to(x.device)
        x = self.encoder(x).permute(0, 2, 1) # (N, C, n_patch)
        x = self.up_nets(x) # (N, C, window_size)
        x = self.detokenlizer(x).squeeze(1) # (N, window_size)
        y = x.relu()
        if self.training:
            return ((gt_apps - y)**2).mean()
        else:
            return y

