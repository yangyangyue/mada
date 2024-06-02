import torch
from torch import nn

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)

class FeatureExtract(nn.Module):
    def __init__(self, in_channels, channels, out_channels, n_layers, fusion,):
        super().__init__()
        self.fusion = fusion
        # conv in
        in_ = in_channels * 2 if self.fusion=='concat' else in_channels
        self.conv_in1 = nn.Conv1d(in_, channels, kernel_size=3, stride=1, padding=1)
        if self.fusion=='cross': self.conv_in2 = nn.Conv1d(in_channels, channels, kernel_size=3, stride=1, padding=1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=channels, nhead=1, dim_feedforward=512, dropout=0.1, batch_first=True)
        encoder_norm = nn.LayerNorm(d_model=channels)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers, encoder_norm)

        # conv out
        self.conv_out = nn.Conv1d(channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, context=None):
        if self.fusion == 'concat': x = torch.concat((x, context), dim=1)
        x = self.conv_in1(x)
        x = self.encoder(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.conv_out(x)
        return x
    
class TmpNet(nn.Module):
    def __init__(self, patch_size=16, patch_stride=16, channels=256, n_layers=6, fusion='concat'):
        super().__init__()
        assert patch_size == patch_stride or patch_size == 2 * patch_stride
        self.fusion = fusion
        dilation = patch_size // patch_stride
        activation = nn.ReLU() if activation is None else activation
        def unfold(tensor: torch.Tensor):
            len_pad =  patch_size - patch_stride
            tensor = nn.functional.pad(tensor, (0, len_pad), mode="constant", value=0)
            tensor = tensor.unfold(dimension=-1, size=patch_size, step=patch_stride)
            return tensor.permute(0, 2, 1)
        self.unfold = unfold
        self.fex = FeatureExtract(patch_size, channels, patch_size, n_layers, fusion)
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
        if self.fusion: context = self.unfold(context)
        # feed x to autoencoder, don't change the shape
        x = self.fex(x, context)
        # fold x and make output using a linear layer
        y = self.fold(x)
        if self.training:
            return ((y-y_hat) ** 2).mean()
        else:
            return y