"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

import torch
from torch import nn

from models.common import Attention, ExampleEncoder, IbnNet



class DownNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ibn = IbnNet(in_channels, out_channels,)

    def forward(self, x):
        x = self.ibn(x)
        return torch.max_pool1d(x, kernel_size=2)


class UpNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ibn = IbnNet(in_channels, out_channels)
        self.attention = Attention(out_channels)
        self.up = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        

    def forward(self, y, x):
        y = self.ibn(y)
        y = y + self.attention(y, x, x)
        y = self.up(y)
        return y

class Encoder(nn.Module):
    def __init__(self, channels, n_layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DownNet(1, channels)] + [DownNet(channels, channels) for _ in range(n_layers-1)])
    
    def forward(self, x):
        xs = []
        for layer in self.layers:
            x = layer(x)
            xs.append(x)
        return xs



class Decoder(nn.Module):
    def __init__(self, channels, n_layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList([UpNet(1, channels)] + [UpNet(channels, channels) for _ in range(n_layers-1)])
    
    def forward(self, y, xs):
        for x, layer in zip(xs, self.layers):
            y = layer(y, x)
        return y
    
class AadaNet(nn.Module):
    def __init__(self, channels, n_layers, window_size=1024) -> None:
        super().__init__()
        self.example_encoder = ExampleEncoder(channels)
        self.encoder = Encoder(channels, n_layers)
        self.combine = Attention()
        self.decoder = Decoder(channels, n_layers)
        feature_length = window_size // (1 << n_layers)
        self.linear = nn.Linear(channels * feature_length, feature_length)
        self.make_out = nn.Conv1d(channels, 1, kernel_size=3, stride=1, padding=1)
        
        

    def forward(self, _, x, gt_apps=None):
        """
        Args:
            examples (N, 3, L): input examples
            samples (N, L): input samples
        """
        xs = self.encoder(x[:, None, :]) # xs: [(N, C, L)] * n_layers
        # combine
        z = xs[-1].flatten(start_dim=1)
        z = self.linear(z)
        # decode
        y = self.decoder(z[:, None, :], reversed(xs))
        y = self.make_out(y)
        y = torch.relu(y).squeeze(1)
        if self.training:
            return ((gt_apps - y)**2).mean()
        else:
            return y

