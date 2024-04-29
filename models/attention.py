import torch
from torch import nn

ONCE_CROSS = True

def softmax_1(tensor, dim=-1):
    exp_tensor = torch.exp(tensor-tensor.max(dim=dim, keepdim=True)[0])
    norm_factor = 1 + exp_tensor.sum(dim=-1, keepdim=True)
    return exp_tensor / norm_factor

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
    
class AttnBlock(nn.Module):
    def __init__(self, channels, softmax):
        super().__init__()
        self.in_channels = channels
        self.softmax = softmax
        self.norm = nn.BatchNorm1d(channels)
        self.q = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x, context=None):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        if context is None: context = h_
        k = self.k(context)
        v = self.v(context)
        # compute attention
        b,c,l = q.shape
        q = q.permute(0,2,1)   # b,l,c
        w_ = torch.bmm(q,k)     # b,l,l   
        w_ = w_ * (int(c)**(-0.5))
        if self.softmax == '_1': w_= softmax_1(w_, dim=2)
        else: w_ = torch.nn.functional.softmax(w_, dim=2)
        # attend to values
        w_ = w_.permute(0,2,1)   # b,l,l (first l of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw 
        h_ = self.proj_out(h_)
        return x + h_
    
class EncoderLayer(nn.Module):
    def __init__(self, channels, conv, attn, cross, softmax, activation):
        super().__init__()
        self.conv, self.attn, self.cross = conv, attn, cross
        if self.conv: self.res1 = ResnetBlock(channels, channels, activation)
        if self.conv and self.cross: self.res2 = ResnetBlock(channels, channels, activation)
        if self.attn: self.self_1 = AttnBlock(channels,softmax)
        if self.attn and self.cross: self.self_2 = AttnBlock(channels, softmax)
        if self.cross: self.cross_attn = AttnBlock(channels, softmax)

    
    def forward(self, x, context=None):
        x = nn.functional.max_pool1d(x, 2)
        if self.cross: context = nn.functional.max_pool1d(context, 2)
        if self.conv: x = self.res1(x)
        if self.conv and self.cross: context = self.res2(context)
        if self.attn: x = self.self_1(x)
        if self.attn and self.cross: context = self.self_2(context)
        if self.cross: x = self.cross_attn(x, context)

        return (x, context) if self.cross else x

class Encoder(nn.Module):
    def __init__(self, i_channels, channels, z_channels, n_layers, conv, attn, fusion, softmax, activation):
        super().__init__()
        self.fusion = fusion
        # conv in
        self.conv_in1 = ResnetBlock(i_channels * 2 if self.fusion=='concat' else i_channels, channels, activation)
        if self.fusion=='cross': self.conv_in2 = nn.Conv1d(i_channels, channels, kernel_size=3, stride=1, padding=1)
        if self.fusion=='cross' and ONCE_CROSS: self.once_cross = AttnBlock(channels, softmax)
        # downsampling

        # downsampling
        self.down = nn.ModuleList([EncoderLayer(channels, conv, attn, self.fusion=='cross' and (not ONCE_CROSS), softmax, activation) for _ in range(n_layers)])
        # conv out
        self.conv_out = nn.Conv1d(channels, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, context=None):
        if self.fusion == 'concat': x = torch.concat((x, context), dim=1)
        x = self.conv_in1(x)
        hs = [x]
        if self.fusion=='cross': context = self.conv_in2(context)
        if self.fusion=='cross' and ONCE_CROSS: x = self.once_cross(x, context)
        for layer in self.down:
            if self.fusion=='cross' and (not ONCE_CROSS): x, context = layer(x, context)
            else: x = layer(x)
            hs.append(x)
        z = self.conv_out(x)
        return z, hs

class DecoderLayer(nn.Module):
    def __init__(self, channels, conv, attn, bridge, softmax, activation):
        super().__init__()
        self.conv, self.attn, self.bridge = conv, attn, bridge
        self.up = nn.ConvTranspose1d((2 * channels) if self.bridge == 'concat' else channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        if self.bridge == 'cross': self.combine = AttnBlock(channels, softmax)
        if self.conv: self.res1 = ResnetBlock(channels, channels, activation)
        if self.attn: self.self_ = AttnBlock(channels, softmax)
       
    
    def forward(self, x, context=None):
        # combine
        if self.bridge == 'cross': x = self.up(self.combine(x, context))
        elif self.bridge == 'concat': x = self.up(torch.cat([x, context], dim=1))
        else: x = self.up(x)
        if self.conv: x = self.res1(x)
        if self.attn: x = self.self_(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, out_channels, channels, z_channels, n_layers, conv, attn, bridge, softmax, activation):
        super().__init__()
        self.bridge=bridge
        self.conv_in = ResnetBlock(z_channels, channels, activation)
        # upsampling
        self.up = nn.ModuleList([DecoderLayer(channels, conv, attn, bridge, softmax, activation) for _ in range(n_layers)])
        # conv out
        self.conv_out = nn.Conv1d(channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z, contexts=None):
        h = self.conv_in(z)
        if self.bridge: 
            for layer, context in zip(self.up, contexts): h = layer(h, context)
        else:
            for layer in self.up: h = layer(h)
        y = self.conv_out(h)
        return y

class AutoEncoder(nn.Module):
    def __init__(self, io_channels, channels, z_channels, n_layers, conv, attn, fusion, bridge, kl, softmax, activation):
        super().__init__()
        self.bridge, self.kl = bridge, kl
        self.encoder = Encoder(io_channels, channels, 2*z_channels if self.kl else z_channels, n_layers, conv, attn, fusion, softmax, activation)
        self.decoder = Decoder(io_channels, channels, z_channels, n_layers, conv, attn, bridge, softmax, activation)
    
    def forward(self, x, context=None):
        # encoder
        z, hs = self.encoder(x, context)
        # do variation inference if kl==True
        if self.kl:
            mu, logvar = torch.chunk(z, 2, dim=1)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps * std + mu
        # decoder
        if self.bridge: y = self.decoder(z, hs[::-1])
        else: y = self.decoder(z)
        return (y, mu, logvar) if self.kl else y