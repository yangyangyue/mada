from torch import nn

from models.ae import AutoEncoder

class AadaNet(nn.Module):
    def __init__(self, channels=256, n_layers=6, conv=True, attn=True, bridge='concat', kl=False):
        super().__init__()
        self.kl = kl
        self.ae = AutoEncoder(1, channels, n_layers, conv, attn, bridge, kl)
    
    def forward(self, x, context=None, y_hat=None):
        if self.kl: x, mu, logvar = self.ae(x[:, None, :], context[:, None, :])
        else: x = self.ae(x[:, None, :], context[:, None, :])
        y = x.squeeze(1)
        if self.training:
            loss = ((y-y_hat) ** 2).mean()
            if self.kl: loss += (-0.5 * (1 + logvar - mu ** 2 - logvar.exp())).sum(dim=(1, 2)).mean()
            return loss
        else:
            return y