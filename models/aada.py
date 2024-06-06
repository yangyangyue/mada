import torch
from torch import nn

from models.attention import AutoEncoder

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)

class AadaNet(nn.Module):
    def __init__(self, channels=256, z_channels=1, n_layers=6, conv=True, attn=False, fusion='concat', bridge='concat', kl=False, softmax='_0', activation=None):
        super().__init__()
        self.fusion, self.kl = fusion, kl
        activation = nn.ReLU() if activation is None else activation
        self.ae = AutoEncoder(channels, z_channels, n_layers, conv, attn, fusion, bridge, kl, softmax, activation)
    
    def forward(self, x, context=None, y_hat=None):
        # unfold x and context to shape (N, patch_size, window_size//patch_stride)
        x = self.unfold(x) 
        if self.fusion: context = self.unfold(context)
        # feed x to autoencoder, don't change the shape
        if self.kl: x, mu, logvar = self.ae(x, context)
        else: x = self.ae(x, context)
        # fold x and make output using a linear layer
        y = self.fold(x)
        if self.training:
            loss = ((y-y_hat) ** 2).mean()
            if self.kl: loss += (-0.5 * (1 + logvar - mu ** 2 - logvar.exp())).sum(dim=(1, 2)).mean()
            return loss
        else:
            return y