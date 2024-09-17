import torch
from torch import nn

from models.ae import AutoEncoder

class AadaNet(nn.Module):
    def __init__(self, channels=256, n_layers=6, n_heads=4, kl=True):
        super().__init__()
        self.kl = kl
        self.ae = AutoEncoder(1, channels, n_layers, n_heads, kl)
    
    def forward(self, ids, x, context=None, y_hat=None, weights=None):
        if self.kl: y, mu, logvar = self.ae(x[:, None, :], context)
        else: y = self.ae(x[:, None, :], context)
        y = y.squeeze(1)
        if self.training:
            loss = (((y-y_hat) ** 2) * weights[:, None]).mean()
            if self.kl: loss += (-0.5 * (1 + logvar - mu ** 2 - logvar.exp())).sum(dim=1).mean()
            return loss
        else:
            return y