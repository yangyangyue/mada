import sys
sys.path.append('..')
from dataset import vars
import torch
from torch import nn

class IbnNet(nn.Module):
    def __init__(self, in_channels, out_channels, use_ins=False):
        super().__init__()
        self.use_ins = use_ins
        self.in_channels, self.out_channels = in_channels, out_channels
        mid_channels = out_channels // 4
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.Conv1d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        if self.use_ins: self.ins = nn.InstanceNorm1d(out_channels)

    def forward(self, x):
        if self.in_channels == self.out_channels: x = x + self.bottleneck(x)
        else: x = self.bottleneck(x)
        if self.use_ins: x = self.ins(x)
        return x.relu()
    
class EncoderLayer(nn.Sequential):
    """
    编码层：最大池化下采样 & 残差卷积 & 残差注意力
    """
    def __init__(self, channels, use_ins=False):
        super().__init__()
        self.add_module('max_pool', nn.MaxPool1d(kernel_size=2, stride=2))
        self.add_module('res', IbnNet(channels, channels, use_ins))

class Encoder(nn.Module):
    """
    编码器： 残差卷积升维 & 编码层*n
    """
    def __init__(self, i_channels, channels, n_layers):
        super().__init__()
        self.conv_in = IbnNet(i_channels , channels, True)
        self.down = nn.ModuleList([EncoderLayer(channels, i<4) for i in range(n_layers)])

    def forward(self, x):
        x = self.conv_in(x)
        return [x] + [(x := layer(x)) for layer in self.down]

class DecoderLayer(nn.Module):
    """
    解码层：cat对应编码器层特征 & 逆卷积上采样 & 残差卷积
    """
    def __init__(self, channels):
        super().__init__()
        self.up = nn.ConvTranspose1d((2 * channels), channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.res = IbnNet(channels, channels)
    
    def forward(self, x, context):
        x = self.up(torch.cat([x, context], dim=1))
        x = self.res(x)
        return x
    
class Decoder(nn.Module):
    """
    解码器： 残差卷积升维 & 解码层*n & 卷积层输出
    """
    def __init__(self, out_channels, channels, n_layers):
        super().__init__()
        # conv in
        self.conv_in = IbnNet(1, channels)
        # upsampling
        self.up = nn.ModuleList([DecoderLayer(channels) for _ in range(n_layers)])
        # conv out
        self.conv_out = nn.Conv1d((2 * channels), out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z, contexts=None):
        h = self.conv_in(z)
        for layer, context in zip(self.up, contexts): h = layer(h, context)
        y = self.conv_out(torch.cat([h, contexts[-1]], dim=1))
        return y

class MadaNet(nn.Module):
    def __init__(self, channels=256, n_layers=6):
        super().__init__()
        self.encoder = Encoder(1, channels, n_layers)
        self.decoder = Decoder(1, channels, n_layers)
        length = vars.WINDOW_SIZE >> n_layers
        self.z_mu = nn.Linear(channels * length + 5, length)
        self.z_var = nn.Linear(channels * length + 5, length)
    
    def forward(self, ids, x, tags=None, y_hat=None, weights=None):
        xs = self.encoder(x[:, None, :])
        x = xs[-1].flatten(start_dim=1)
        x = torch.concat([tags, x], dim=1)
        z = (mu := self.z_mu(x)) + (std := torch.exp(0.5 * (logvar := self.z_var(x)))) * torch.randn_like(std)
        y = self.decoder(z[:, None, :], xs[::-1])
        y = y.relu().squeeze(1)
        if self.training:
            loss = (((y-y_hat) ** 2) * weights[:, None]).mean()
            loss += (-0.5 * (1 + logvar - mu ** 2 - logvar.exp())).sum(dim=1).mean()
            return loss
        else:
            return y
