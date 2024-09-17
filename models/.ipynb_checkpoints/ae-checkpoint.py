import torch
from torch import nn

WINDOW_SIZE = 1024

class ResnetBlock(nn.Module):
    """
    瓶颈残差块：减少参数量 & 增加网络深度
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = max(max(in_channels, out_channels) // 4, 1)
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
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        return (self.shortcut(x) + self.bottleneck(x)).relu()

class AttnBlock(nn.Module):
    """
    残差多头注意力块：context为空时，对x计算self-attention，否则以context为kv计算cross-attention
    """
    def __init__(self, d_model, n_heads) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.wq = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.wk = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.wv = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.out = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)

    def forward(self, x, context=None):
        q, k, v = (x, x, x) if context is None else (x, context, context)
        N = q.shape[0]
        q = self.wq(q).reshape([N, self.n_heads, self.d_head, -1])
        k = self.wk(k).reshape([N, self.n_heads, self.d_head, -1])
        v = self.wv(v).reshape([N, self.n_heads, self.d_head, -1])

        atten = torch.einsum('nhdq,nhdk->nhqk', q, k)
        atten = atten / (self.d_model ** 0.5)
        atten = torch.softmax(atten, dim=-1)

        v = torch.einsum('nhqk,nhdk->nhdq', atten, v).reshape([N, self.d_model, -1])
        return x + self.out(v)
    
class EncoderLayer(nn.Sequential):
    """
    编码层：最大池化下采样 & 残差卷积 & 残差注意力
    """
    def __init__(self, channels, n_heads):
        super().__init__()
        self.add_module('max_pool', nn.MaxPool1d(kernel_size=2, stride=2))
        self.add_module('res', ResnetBlock(channels, channels))
        # self.add_module('msa', AttnBlock(channels, n_heads))

class Encoder(nn.Module):
    """
    编码器： 残差卷积升维 & 编码层*n
    """
    def __init__(self, i_channels, channels, n_layers, n_heads):
        super().__init__()
        self.conv_in = ResnetBlock(i_channels , channels)
        self.down = nn.ModuleList([EncoderLayer(channels, n_heads) for _ in range(n_layers)])

    def forward(self, x):
        return [x := self.conv_in(x)] + [x := layer(x) for layer in self.down]

class DecoderLayer(nn.Module):
    """
    解码层：cat对应编码器层特征 & 逆卷积上采样 & 残差卷积
    此处不使用attention是认为attention逻辑上应该是用于特征抽取，解码时上采样即可
    """
    def __init__(self, channels):
        super().__init__()
        self.up = nn.ConvTranspose1d((2 * channels), channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.res = ResnetBlock(channels, channels)
    
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
        self.conv_in = ResnetBlock(1, channels)
        # upsampling
        self.up = nn.ModuleList([DecoderLayer(channels) for _ in range(n_layers)])
        # conv out
        self.conv_out = nn.Conv1d((2 * channels), out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z, contexts=None):
        h = self.conv_in(z)
        for layer, context in zip(self.up, contexts): h = layer(h, context)
        y = self.conv_out(torch.cat([h, contexts[-1]], dim=1))
        return y

class AutoEncoder(nn.Module):
    """
    自编码器：kl为true则会进行KL散度处理
    """
    def __init__(self, io_channels, channels, n_layers, n_heads, kl):
        super().__init__()
        self.kl = kl
        self.encoder = Encoder(io_channels, channels, n_layers, n_heads)
        self.decoder = Decoder(io_channels, channels, n_layers)
        length = WINDOW_SIZE >> n_layers
        append = 5
        self.z_mu = nn.Linear(channels * length + append, length)
        if self.kl: self.z_var = nn.Linear(channels * length + append, length)
    
    def forward(self, x, context):
        # encoder
        xs = self.encoder(x)
        x = xs[-1].flatten(start_dim=1)
        x = torch.concat([x, context], dim=1)
        z = (mu := self.z_mu(x))
        # kl
        if self.kl: z += (std := torch.exp(0.5 * (logvar := self.z_var(x)))) * torch.randn_like(std)
        # decoder
        y = self.decoder(z[:, None, :], xs[::-1])
        return (y, mu, logvar) if self.kl else y
