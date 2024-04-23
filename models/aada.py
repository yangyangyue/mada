import torch
from torch import nn

from models.attention import AutoEncoder

class AadaNet(nn.Module):
    def __init__(self, window_size=1024, patch_size=8, patch_stride=4, channels_list=(64, 128, 256), conv=True, attn=True, cross=True, bridge='cross', kl=True, activation=None):
        self.cross, self.kl = cross, kl
        dilation = patch_size // patch_stride
        activation = nn.ReLU() if activation is None else activation
        def unfold(tensor: torch.Tensor):
            len_pad =  patch_size - patch_stride
            tensor = nn.functional.pad(tensor, (0, len_pad), mode="constant", value=0)
            tensor = tensor.unfold(dimension=-1, size=patch_size, step=patch_stride)
            return tensor.permute(0, 2, 1)
        self.unfold = unfold
        self.ae = AutoEncoder(patch_size, channels_list, patch_size, conv, attn, cross, bridge, kl, activation)
        self.fold = nn.Sequential(lambda tensor: tensor.permute(0, 2, 1), nn.Flatten(), nn.Linear(dilation * window_size, window_size), activation)
    
    def forward(self, x, context=None, y_hat=None):
        # unfold x: patch_size patch_num
        x = self.unfold(x) 
        if self.cross: context = self.unfold(context)
        # auto encode x: N patch_size patch_num
        if self.kl: x, mu, logvar = self.ae(x, context)
        else: x = self.ae(x, context)
        # fold
        y = self.fold(x)
        if self.training:
            loss = ((y-y_hat) ** 2).mean()
            if self.kl: loss += (-0.5 * (1 + logvar - mu ** 2 - logvar.exp())).sum(dim=(1, 2)).mean()
            return loss
        else:
            return y