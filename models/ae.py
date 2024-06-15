import torch
from torch import nn

def softmax_1(tensor, dim=-1):
    exp_tensor = torch.exp(tensor-tensor.max(dim=dim, keepdim=True)[0])
    norm_factor = 1 + exp_tensor.sum(dim=-1, keepdim=True)
    return exp_tensor / norm_factor

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = max(max(in_channels, out_channels) // 4, 1)
        self.stream = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.Conv1d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
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
        w_ = torch.nn.functional.softmax(w_, dim=2)
        # attend to values
        w_ = w_.permute(0,2,1)   # b,l,l (first l of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw 
        h_ = self.proj_out(h_)
        return x + h_
    
class EncoderLayer(nn.Module):
    def __init__(self, channels, conv, attn, fusion, softmax):
        super().__init__()
        self.conv, self.attn, self.cross = conv, attn, fusion=='cross'
        if self.conv: self.res1 = ResnetBlock(channels, channels)
        if self.attn: self.self_1 = AttnBlock(channels, softmax)
        if self.cross: 
            self.res2 = ResnetBlock(channels, channels)
            self.cross_attn = AttnBlock(channels, softmax)
    
    def forward(self, x, context=None):
        x = nn.functional.max_pool1d(x, 2)
        if self.conv: x = self.res1(x)
        if self.attn: x = self.self_1(x)
        if self.cross: 
            context = nn.functional.max_pool1d(context, 2)
            context = self.res2(context)
            x = self.cross_attn(x, context)
        return (x, context) if self.cross else x

class Encoder(nn.Module):
    def __init__(self, i_channels, channels, n_layers, conv, attn, fusion, softmax):
        super().__init__()
        self.fusion = fusion
        # conv in
        self.conv_in1 = ResnetBlock(i_channels * 2 if self.fusion=='concat' else i_channels, channels)
        if self.fusion=='cross': self.conv_in2 = ResnetBlock(i_channels, channels)
        # downsampling
        self.down = nn.ModuleList([EncoderLayer(channels, conv, attn, self.fusion, softmax) for _ in range(n_layers)])

    def forward(self, x, context=None):
        if self.fusion == 'concat': x = torch.concat((x, context), dim=1)
        x = self.conv_in1(x)
        hs = [x]
        if self.fusion=='cross': context = self.conv_in2(context)
        for layer in self.down:
            if self.fusion=='cross': x, context = layer(x, context)
            else: x = layer(x)
            hs.append(x)
        return hs

class DecoderLayer(nn.Module):
    def __init__(self, channels, bridge, softmax):
        super().__init__()
        self.bridge = bridge
        self.up = nn.ConvTranspose1d((2 * channels) if self.bridge == 'concat' else channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        if self.bridge == 'cross': self.combine = AttnBlock(channels, softmax)
        self.res1 = ResnetBlock(channels, channels)
    
    def forward(self, x, context=None):
        # combine
        if self.bridge == 'cross': x = self.up(self.combine(x, context))
        elif self.bridge == 'concat': x = self.up(torch.cat([x, context], dim=1))
        else: x = self.up(x)
        x = self.res1(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, out_channels, channels, n_layers, bridge, softmax):
        super().__init__()
        self.bridge=bridge
        # conv in
        self.conv_in = ResnetBlock(1, channels)
        # upsampling
        self.up = nn.ModuleList([DecoderLayer(channels, bridge, softmax) for _ in range(n_layers)])
        # conv out
        self.conv_out = nn.Conv1d((2 * channels) if self.bridge == 'concat' else channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z, contexts=None):
        h = self.conv_in(z)
        if self.bridge: 
            for layer, context in zip(self.up, contexts): h = layer(h, context)
            y = self.conv_out(torch.cat([h, contexts[-1]], dim=1))
        else:
            for layer in self.up: h = layer(h)
            y = self.conv_out(h)
        return y

class AutoEncoder(nn.Module):
    def __init__(self, io_channels, channels, n_layers, conv, attn, fusion, bridge, kl, softmax):
        super().__init__()
        self.bridge, self.kl = bridge, kl
        self.encoder = Encoder(io_channels, channels, n_layers, conv, attn, fusion, softmax)
        self.decoder = Decoder(io_channels, channels, n_layers, bridge, softmax)
        length = 1024 // (1 << n_layers)
        self.z_mu = nn.Linear(channels * length, length)
        if self.kl: self.z_log_var = nn.Linear(channels * length, length)
    
    def forward(self, x, context=None):
        # encoder
        hs = self.encoder(x, context)
        # do variation inference if kl==True
        mid = hs[-1].flatten(start_dim=1)
        z = self.z_mu(mid)
        if self.kl:
            logvar = self.z_log_var(mid)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps * std + z
        # decoder
        if self.bridge: y = self.decoder(z[:, None, :], hs[::-1])
        else: y = self.decoder(z[:, None, :])
        return (y, z, logvar) if self.kl else y