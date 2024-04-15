"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

import math
import torch
from torch import Tensor, nn

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
    
class PtNet(nn.Module):
    def __init__(self, channels=256, n_layers=2, window_size=1024) -> None:
        super().__init__()
        self.tokenlizer = nn.Linear(16, channels)
        encoder_layer = nn.TransformerEncoderLayer(channels, 8, 1048, batch_first=True)
        norm = nn.Sequential(nn.Transpose(1,2), nn.BatchNorm1d(channels), nn.Transpose(1,2))
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers, norm)
        self.linear = nn.Linear(2 * channels * window_size // 16, window_size)

    def forward(self, e: Tensor, x: Tensor, gt_apps:Tensor=None):
        """
        Args:
            examples (N, 3, L): input examples
            samples (N, L): input samples
        """
        x = x.unfold(dimension=-1, size=16, step=16)
        e = e.unfold(dimension=-1, size=16, step=16)
        x = torch.cat([x, e], dim=1)
        x = self.tokenlizer(x)
        x = x + pe(x.shape[1], x.shape[2])
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        y = x.relu()
        if self.training:
            return ((gt_apps - y)**2).mean()
        else:
            return y

