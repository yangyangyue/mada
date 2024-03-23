"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

import sys
sys.path.append('/home/aistudio/external-libraries')

import torch
from torch import nn

USE_ATTENTION = True

class ResBlock(nn.Module):
    def __init__(self, inplates, midplates) -> None:
        super().__init__()
        self.stream = nn.Sequential(
            nn.Conv1d(in_channels=inplates, out_channels=midplates, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(midplates),
            nn.ReLU(),
            nn.Conv1d(in_channels=midplates, out_channels=midplates,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(midplates),
            nn.ReLU(),
            nn.Conv1d(in_channels=midplates, out_channels=inplates, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(inplates)
        )
        self.norm = nn.InstanceNorm1d(256)
    def forward(self, x):
        x = x + self.stream(x)
        return torch.relu(self.norm(x))

class DownSampleNetwork(nn.Module):
    def __init__(self, inplates, midplates):
        super().__init__()
        self.res = ResBlock(inplates, midplates)

    def forward(self, x):
        x = self.res(x)
        return torch.max_pool1d(x, kernel_size=2)


class UpSampleNetwork(nn.Module):
    def __init__(self, inplates, midplates):
        super().__init__()
        self.up_sampler = nn.ConvTranspose1d(in_channels=inplates, out_channels=inplates, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.res = ResBlock(inplates, midplates)

    def forward(self, x):
        x = self.up_sampler(x)
        x = self.res(x)
        return x
    
class PositionEmbeddingSine(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        inplates, length = x.shape[1], x.shape[2]
        pe = torch.zeros(length, inplates) 
        position = torch.arange(0, length)
        div_term = torch.full([1, inplates // 2], 10000).pow((torch.arange(0, inplates, 2) / inplates))
        pe[:, 0::2] = torch.sin(position[:, None] / div_term)
        pe[:, 1::2] = torch.cos(position[:, None] / div_term)
        return self.dropout(pe.permute(1, 0).to(x.device))
    

class Attention(nn.Module):
    def __init__(self, d_model, n_heads) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q,k,v):
        N = q.shape[0]
        q = self.wq(q).reshape([N, -1, self.n_heads, self.d_head])
        k = self.wk(k).reshape([N, -1, self.n_heads, self.d_head])
        v = self.wv(v).reshape([N, -1, self.n_heads, self.d_head])

        atten = torch.einsum('nqhd,nkhd->nhqk', q, k)
        atten = atten / (self.d_model ** 0.5)
        atten = torch.softmax(atten, dim=-1)

        v = torch.einsum('nhqk,nkhd->nqhd', atten, v).reshape([N, -1, self.d_model])
        return self.out(v)
    
class ExampleEncoderLayer(nn.Module):
    def __init__(self, inplates, midplates, n_heads, dropout) -> None:
        super().__init__()
        self.down_sampler = DownSampleNetwork(inplates, midplates)
        if USE_ATTENTION:
            self.attention = Attention(inplates, n_heads)
            self.norm = nn.BatchNorm1d(inplates)
            self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x (N, D, L): the features of examples to each encoder layer
        """
        x = self.down_sampler(x)
        if USE_ATTENTION:
            x = self.norm(x).permute(0, 2, 1)
            x = x + self.dropout(self.attention(x, x, x))
            return x.permute(0, 2, 1)
        return x
    

class SampleEncoderLayer(nn.Module):
    def __init__(self, inplates, midplates, n_heads, dropout) -> None:
        super().__init__()
        self.down_sampler = DownSampleNetwork(inplates, midplates)
        if USE_ATTENTION:
            self.attention1 = Attention(inplates, n_heads)
            self.norm1 = nn.BatchNorm1d(inplates)
            self.dropout1 = nn.Dropout(dropout)
        self.attention2 = Attention(inplates, n_heads)
        self.norm2 = nn.BatchNorm1d(inplates)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, y):
        """
        Args:
            x (N, D, L): the features of samples
            y (N, D, L): the features of examples
        """
        y = y.permute(0, 2, 1)
        x = self.down_sampler(x)
        if USE_ATTENTION:
            x = self.norm1(x).permute(0, 2, 1)
            x = x + self.dropout1(self.attention1(x, x, x))
            x = x.permute(0, 2, 1)
        x = self.norm2(x).permute(0, 2, 1)
        x = x + self.dropout2(self.attention2(x, y, y))
        return x.permute(0, 2, 1)
    
class DecoderLayer(nn.Module):
    def __init__(self, inplates, midplates, n_heads, dropout) -> None:
        super().__init__()
        self.up_sampler = UpSampleNetwork(inplates, midplates)
        if USE_ATTENTION:
            self.attention1 = Attention(inplates, n_heads)
            self.norm1 = nn.BatchNorm1d(inplates)
            self.dropout1 = nn.Dropout(dropout)
        self.attention2 = Attention(inplates, n_heads)
        self.norm2 = nn.BatchNorm1d(inplates)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, y):
        """
        Args:
            x (N, D, L): the features of decoder
            y (N, D, L): the features of samples
        """
        y = y.permute(0, 2, 1)
        x = self.up_sampler(x)
        if USE_ATTENTION:
            x = self.norm1(x).permute(0, 2, 1)
            x = x + self.dropout1(self.attention1(x, x, x))
            x = x.permute(0, 2, 1)
        x = self.norm2(x).permute(0, 2, 1)
        x = x + self.dropout2(self.attention2(x, y, y))
        return x.permute(0, 2, 1)
    

class ExampleEncoder(nn.Module):
    def __init__(self, inplates, midplates, n_heads, dropout, n_layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ExampleEncoderLayer(inplates, midplates, n_heads, dropout) for _ in range(n_layers)])
    
    def forward(self, x):
        examples = []
        for layer in self.layers:
            x = layer(x)
            examples.append(x)
        return examples

class SampleEncoder(nn.Module):
    def __init__(self, inplates, midplates, n_heads, dropout, n_layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList([SampleEncoderLayer(inplates, midplates, n_heads, dropout) for _ in range(n_layers)])
    
    def forward(self, x, examples):
        samples = []
        for example, layer in zip(examples, self.layers):
            x = layer(x, example)
            samples.append(x)
        return samples
    
class Decoder(nn.Module):
    def __init__(self, inplates, midplates, n_heads, dropout, n_layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(inplates, midplates, n_heads, dropout) for _ in range(n_layers)])
    
    def forward(self, x, samples):
        for sample, layer in zip(samples, self.layers):
            x = layer(x, sample)
        return x
    
class AadaNet(nn.Module):
    def __init__(self, inplates, midplates, n_heads, dropout, n_layers) -> None:
        super().__init__()
        self.up_dim1 = nn.Conv1d(in_channels=1, out_channels=inplates, kernel_size=3, stride=1, padding=1)
        self.up_dim2 = nn.Conv1d(in_channels=1, out_channels=inplates, kernel_size=3, stride=1, padding=1)
        self.res1 = ResBlock(inplates, midplates)
        self.res2 = ResBlock(inplates, midplates)
        self.down_dim = nn.Conv1d(in_channels=inplates, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.pe = PositionEmbeddingSine()

        self.example_encoder = ExampleEncoder(inplates, midplates, n_heads, dropout, n_layers)
        self.sample_encoder = SampleEncoder(inplates, midplates, n_heads, dropout, n_layers)
        self.decoder = Decoder(inplates, midplates, n_heads, dropout, n_layers)

    def forward(self, examples, samples, gt_apps=None):
        """
        Args:
            examples (N, L): input examples
            samples (N, L): input samples
        """
        # examples | samples: (N, D, L)
        examples = self.up_dim1(examples[:, None, :])
        samples = self.up_dim2(samples[:, None, :])
        examples = examples + self.pe(examples)
        samples = samples + self.pe(samples)

        examples = self.example_encoder(examples)
        samples = self.sample_encoder(samples, examples)
        features = self.res1(samples[-1])
        pred_apps = self.decoder(features, reversed(samples))
        pred_apps = self.res2(pred_apps)
        pred_apps = self.down_dim(pred_apps)
        pred_apps = torch.relu(pred_apps).squeeze(1)
        if self.training:
            return ((gt_apps - pred_apps)**2).mean()
        else:
            return pred_apps

    
    
