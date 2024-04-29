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
    def __init__(self, patch_size=1, patch_stride=1, channels=256, z_channels=1, n_layers=6, conv=True, attn=False, cross=False, bridge='concat', kl=False, softmax='_0', activation=None):
        super().__init__()
        assert patch_size == patch_stride or patch_size == 2 * patch_stride
        self.cross, self.kl = cross, kl
        dilation = patch_size // patch_stride
        activation = nn.ReLU() if activation is None else activation
        def unfold(tensor: torch.Tensor):
            len_pad =  patch_size - patch_stride
            tensor = nn.functional.pad(tensor, (0, len_pad), mode="constant", value=0)
            tensor = tensor.unfold(dimension=-1, size=patch_size, step=patch_stride)
            return tensor.permute(0, 2, 1)
        self.unfold = unfold
        self.ae = AutoEncoder(patch_size, channels, z_channels, n_layers, conv, attn, cross, bridge, kl, softmax, activation)
        if dilation == 1:
            self.fold = nn.Sequential(Lambda(lambda tensor: tensor.permute(0, 2, 1)), nn.Flatten(), nn.ReLU())
        else:
            self.fold = nn.Sequential(
                nn.Conv1d(patch_size, patch_size, kernel_size=3, stride=1, padding=1), 
                nn.MaxPool1d(2), 
                nn.Conv1d(patch_size, patch_size, kernel_size=3, stride=1, padding=1), 
                Lambda(lambda tensor: tensor.permute(0, 2, 1)), 
                nn.Flatten(), 
                nn.ReLU())
    
    def forward(self, x, context=None, y_hat=None):
        # unfold x and context to shape (N, patch_size, window_size//patch_stride)
        x = self.unfold(x) 
        if self.cross: context = self.unfold(context)
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