"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""
"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

import math
import torch
from torch import Tensor, nn

from models.transformer import TransformerEncoder




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
    
class AptNet(nn.Module):
    def __init__(self, channels=256, n_layers=4, window_size=1024) -> None:
        super().__init__()
        self.tokenlizer = nn.Linear(16, channels)
        self.encoder = TransformerEncoder(n_layers)
        self.detokenlizer = nn.Linear(channels, 16)
        self.linear = nn.Linear(2 * window_size, window_size)

    def forward(self, e, x: Tensor, gt_apps:Tensor=None):
        """
        Args:
            examples (N, 3, L): input examples
            samples (N, L): input samples
        """
        x = x.unfold(dimension=-1, size=16, step=16)
        e = e.unfold(dimension=-1, size=16, step=16)
        x = torch.cat([x, e], dim=1)
        x = self.tokenlizer(x)
        x = x + pe(x.shape[1], x.shape[2]).to(x.device)
        x = self.encoder(x)
        x = self.detokenlizer(x)
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        y = x.relu()
        if self.training:
            return ((gt_apps - y)**2).mean()
        else:
            return y