import torch
from torch import nn

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stream = nn.Sequential(    
            nn.BatchNorm1d(in_channels),
            activation,
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            activation,
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        if self.in_channels != self.out_channels: self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = self.stream(x)
        if self.in_channels != self.out_channels: x = self.shortcut(x)
        return x + h
    
class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = nn.functional.pad(x, (0, 1), mode="constant", value=0)
        x = self.conv(x)
        return x
    
class AttnBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.in_channels = channels
        self.norm = nn.BatchNorm1d(channels)
        self.q = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x, context=None):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        if not context: context = h_
        k = self.k(context)
        v = self.v(context)
        # compute attention
        b,c,l = q.shape
        q = q.permute(0,2,1)   # b,l,c
        w_ = torch.bmm(q,k)     # b,l,l   
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)
        # attend to values
        w_ = w_.permute(0,2,1)   # b,l,l (first l of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw 
        h_ = self.proj_out(h_)
        return x + h_
    
class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, conv, attn, cross, activation):
        super().__init__()
        self.conv, self.attn, self.cross = conv, attn, cross
        if self.conv: self.res1 = ResnetBlock(in_channels, out_channels, activation)
        if self.conv and self.cross: self.res2 = ResnetBlock(in_channels, out_channels, activation)
        if self.conv: in_channels = out_channels
        if self.attn: self.self_1 = AttnBlock(in_channels)
        if self.attn and self.cross: self.self_2 = AttnBlock(in_channels)
        if self.cross: self.cross_attn = AttnBlock(in_channels)
        self.down1 = Downsample(in_channels)
        if self.cross: self.down2 = Downsample(in_channels)
    
    def forward(self, x, context=None):
        if self.conv: x = self.res1(x)
        if self.conv and self.cross: context = self.res2(context)
        if self.attn: x = self.self_1(x)
        if self.attn and self.cross: context = self.self_2(context)
        if self.cross: x = self.cross_attn(x, context)
        x = self.down1(x)
        if self.cross: context = self.down2(context)
        return x, context if self.cross else x

class Encoder(nn.Module):
    def __init__(self, i_channels, channels_list, z_channels, conv, attn, cross, activation):
        super().__init__()
        self.cross = cross
        # conv in
        self.conv_in1 = nn.Conv1d(i_channels, channels_list[0],  kernel_size=3, stride=1, padding=1)
        if self.cross: self.conv_in2 = nn.Conv1d(i_channels, channels_list[0],  kernel_size=3, stride=1, padding=1)
        # downsampling
        in_channels_list = channels_list[:-1]
        out_channels_list = channels_list[1:]
        self.down = nn.ModuleList([EncoderLayer(in_, out_, conv, attn, cross, activation) for in_, out_ in zip(in_channels_list, out_channels_list)])
        channels = out_channels_list[-1]
        # post
        self.post = nn.Sequential(ResnetBlock(channels, channels, activation), AttnBlock(channels), ResnetBlock(channels, channels, activation))
        # conv out
        self.conv_out = nn.Sequential(nn.BatchNorm1d(), activation, nn.Conv1d(channels, z_channels, kernel_size=3, stride=1, padding=1))

    def forward(self, x, context=None):
        x = self.conv_in1(x)
        hs = [x]
        if self.cross: context = self.conv_in2(context)
        for layer in self.down:
            if self.cross: x, context = layer(x, context)
            else: x = layer(x)
            hs.append(x)
        h = self.post(hs[-1])
        z = self.conv_out(h)
        return z, hs

class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, conv, attn, bridge, activation):
        super().__init__()
        self.conv, self.attn, self.bridge = conv, attn, bridge
        if self.conv: 
            self.res1 = ResnetBlock(in_channels, out_channels, activation)
            in_channels = out_channels
        if self.attn: self.self_ = AttnBlock(in_channels)
        if self.bridge == 'cross': self.combine = AttnBlock(in_channels)
        elif self.bridge == 'concat': self.combine = ResnetBlock(in_channels*2, in_channels, activation)
        self.up = Upsample(in_channels)
    
    def forward(self, x, context=None):
        x = self.res1(x)
        x = self.self_(x)
        if self.bridge == 'cross': x = self.combine(x, context)
        elif self.bridge == 'concat': x = self.combine(torch.cat([x, context], dim=1))
        x = self.up(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, out_channels, channels_list, z_channels, conv, attn, bridge, activation):
        super().__init__()
        self.bridge=bridge
        channels = channels_list[-1]
        self.conv_in = nn.Conv1d(z_channels, channels, kernel_size=3, stride=1, padding=1)
        self.pre = nn.Sequential(ResnetBlock(channels, channels, activation), AttnBlock(channels), ResnetBlock(channels, channels, activation))
        # upsampling
        in_channels_list = channels_list[:-1]
        out_channels_list = channels_list[1:]
        self.up = nn.ModuleList([DecoderLayer(in_, out_, conv, attn, bridge, activation) for in_, out_ in zip(in_channels_list, out_channels_list)])
        channels = out_channels_list[-1]
        # post
        self.post = nn.Sequential(ResnetBlock(channels, channels, activation), AttnBlock(channels), ResnetBlock(channels, channels, activation))
        # conv out
        self.conv_out = nn.Sequential(nn.BatchNorm1d(), activation,nn.Conv1d(channels, out_channels, kernel_size=3, stride=1, padding=1))

    def forward(self, z, contexts=None):
        h = self.conv_in(z)
        h = self.pre(h)
        if self.bridge: 
            for layer, context in zip(self.up, contexts): h = layer(h, context)
        else:
            for layer in self.up: h = layer(h)
        y = self.conv_out(h)
        return y

class AutoEncoder(nn.Module):
    def __init__(self, io_channels, channels_list, z_channels, conv, attn, cross, bridge, kl, activation):
        super().__init__()
        self.bridge, self.kl = bridge, kl
        self.encoder = Encoder(io_channels, channels_list, 2*z_channels if self.kl else z_channels, conv, attn, cross, activation)
        self.decoder = Decoder(io_channels, channels_list, z_channels, conv, attn, bridge, activation)
    
    def forward(self, x, context=None):
        z, hs = self.encoder(x, context)
        mu, logvar = torch.chunk(z, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu

        if self.bridge: y = self.decoder(z, hs)
        else: y = self.decoder(z)

        if self.kl: return y, mu, logvar
        else: return y