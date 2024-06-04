import torch
from torch import nn

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)
    
class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = out_channels // 4
        self.stream = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(mid_channels),
            activation,
            nn.Conv1d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(mid_channels),
            activation,
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        h = self.stream(x)
        if self.in_channels == self.out_channels: h += x
        return h
    
class UpNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.res = ResnetBlock(channels, channels, nn.ReLU())
    def forward(self, x):
        x = self.up(x)
        return self.res(x)



class FeatureExtract(nn.Module):
    def __init__(self, in_channels, channels, n_layers, fusion,):
        super().__init__()
        self.fusion = fusion
        # conv in
        in_ = in_channels * 2 if self.fusion=='concat' else in_channels
        self.conv_in1 = nn.Linear(in_, channels)

        encoder_layer = nn.TransformerEncoderLayer(d_model=channels, nhead=1, dim_feedforward=512, dropout=0.1, batch_first=True)
        encoder_norm = nn.LayerNorm(channels)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers, encoder_norm)

        # conv out
        self.conv_out = nn.Sequential(
            UpNet(channels),
            UpNet(channels),
            UpNet(channels),
            UpNet(channels),
            nn.Conv1d(channels, 1, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
            nn.ReLU()
        )

    def forward(self, x, context=None):
        x = self.conv_in1(x)
        x = self.encoder(x).permute(0, 2, 1)
        x = self.conv_out(x)
        return x
    
class TmpNet(nn.Module):
    def __init__(self, patch_size=16, patch_stride=16, channels=256, n_layers=6, fusion='concat'):
        super().__init__()
        assert patch_size == patch_stride or patch_size == 2 * patch_stride
        self.fusion = fusion
        dilation = patch_size // patch_stride
        def unfold(tensor: torch.Tensor):
            len_pad =  patch_size - patch_stride
            tensor = nn.functional.pad(tensor, (0, len_pad), mode="constant", value=0)
            tensor = tensor.unfold(dimension=-1, size=patch_size, step=patch_stride)
            return tensor
        self.unfold = unfold
        self.fex = FeatureExtract(patch_size, channels, n_layers, fusion)
    
    def forward(self, x, context=None, y_hat=None):
        # unfold x and context to shape (N, patch_size, window_size//patch_stride)
        x = self.unfold(x) 
        if self.fusion: context = self.unfold(context)
        # feed x to autoencoder, don't change the shape
        y = self.fex(x, context)
        # fold x and make output using a linear layer
        if self.training:
            return ((y-y_hat) ** 2).mean()
        else:
            return y