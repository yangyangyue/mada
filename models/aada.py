"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

import torch
from torch import nn

from models.ibn import IbnNet


class DownNet(nn.Module):
    def __init__(self, channels, mid_channels, use_ins=True):
        super().__init__()
        self.ibn = IbnNet(channels, mid_channels, channels, use_ins)

    def forward(self, x):
        x = self.ibn(x)
        return torch.max_pool1d(x, kernel_size=2)


class UpNet(nn.Module):
    def __init__(self, channels, mid_channels, use_ins=True):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.ibn = IbnNet(channels, mid_channels, channels, use_ins)

    def forward(self, x):
        x = self.up(x)
        x = self.ibn(x)
        return x
    
class PositionEmbeddingSine(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        channels, length = x.shape[1], x.shape[2]
        pe = torch.zeros(length, channels) 
        position = torch.arange(0, length)
        div_term = torch.full([1, channels // 2], 10000).pow((torch.arange(0, channels, 2) / channels))
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
    def __init__(self, self_attention, channels, n_heads, dropout, mid_channels, use_ins) -> None:
        super().__init__()
        self.self_attention = self_attention
        if self.self_attention:
            self.attention = Attention(channels, n_heads)
            self.norm = nn.BatchNorm1d(channels)
            self.dropout = nn.Dropout(dropout)
        self.down_net = DownNet(channels, mid_channels, use_ins)
    
    def forward(self, exampls):
        """
        Args:
            x (N, D, L): the features of examples to each encoder layer
        """
        if self.self_attention:
            exampls = self.norm(exampls).permute(0, 2, 1)
            exampls = exampls + self.dropout(self.attention(exampls, exampls, exampls))
            exampls = exampls.permute(0, 2, 1)
        exampls = self.down_net(exampls)
        return exampls
    

class SampleEncoderLayer(nn.Module):
    def __init__(self, self_attention, channels, n_heads, dropout, mid_channels, use_ins) -> None:
        super().__init__()
        self.self_attention = self_attention
        if self.self_attention:
            self.attention1 = Attention(channels, n_heads)
            self.norm1 = nn.BatchNorm1d(channels)
            self.dropout1 = nn.Dropout(dropout)
        self.attention2 = Attention(channels, n_heads)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout2 = nn.Dropout(dropout)
        self.down_net = DownNet(channels, mid_channels, use_ins)
    
    def forward(self, samples, examples):
        """
        Args:
            x (N, D, L): the features of samples
            y (N, D, L): the features of examples
        """
        if self.self_attention:
            samples = self.norm1(samples).permute(0, 2, 1)
            samples = samples + self.dropout1(self.attention1(samples, samples, samples))
            samples = samples.permute(0, 2, 1)
        # cross attention
        examples = examples.permute(0, 2, 1)
        samples = self.norm2(samples).permute(0, 2, 1)
        samples = samples + self.dropout2(self.attention2(samples, examples, examples))
        samples = samples.permute(0, 2, 1)
        # down
        samples = self.down_net(samples)
        return samples
    
class DecoderLayer(nn.Module):
    def __init__(self, self_attention, channels, n_heads, dropout, mid_channels, use_ins) -> None:
        super().__init__()
        self.self_attention = self_attention
        if self.self_attention:
            self.attention1 = Attention(channels, n_heads)
            self.norm1 = nn.BatchNorm1d(channels)
            self.dropout1 = nn.Dropout(dropout)
        self.attention2 = Attention(channels, n_heads)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout2 = nn.Dropout(dropout)
        self.up_net = UpNet(channels, mid_channels, use_ins)
    
    def forward(self, preds, samples):
        """
        Args:
            x (N, D, L): the features of decoder
            y (N, D, L): the features of samples
        """
        if self.self_attention:
            preds = self.norm1(preds).permute(0, 2, 1)
            preds = preds + self.dropout1(self.attention1(preds, preds, preds))
            preds = preds.permute(0, 2, 1)
        samples = samples.permute(0, 2, 1)
        preds = self.norm2(preds).permute(0, 2, 1)
        preds = preds + self.dropout2(self.attention2(preds, samples, samples))
        preds = preds.permute(0, 2, 1)
        preds = self.up_net(preds)
        return preds

class ExampleEncoder(nn.Module):
    def __init__(self, self_attention, channels, n_heads, dropout, mid_channels, use_ins, n_layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ExampleEncoderLayer(self_attention, channels, n_heads, dropout, mid_channels, use_ins) 
                                     for _ in range(n_layers)])
    
    def forward(self, x):
        examples = []
        for layer in self.layers:
            x = layer(x)
            examples.append(x)
        return examples

class SampleEncoder(nn.Module):
    def __init__(self, self_attention, channels, n_heads, dropout, mid_channels, use_ins, n_layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList([SampleEncoderLayer(self_attention, channels, n_heads, dropout, mid_channels, use_ins) for _ in range(n_layers)])
    
    def forward(self, x, examples):
        samples = []
        for example, layer in zip(examples, self.layers):
            x = layer(x, example)
            samples.append(x)
        return samples
    
class Decoder(nn.Module):
    def __init__(self, self_attention, channels, n_heads, dropout, mid_channels, use_ins, n_layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(self_attention, channels, n_heads, dropout, mid_channels, use_ins) for _ in range(n_layers)])
    
    def forward(self, x, samples):
        for sample, layer in zip(samples, self.layers):
            x = layer(x, sample)
        return x
    
class AadaNet(nn.Module):
    def __init__(self, self_attention, channels, n_heads, dropout, mid_channels, use_ins, n_layers, variation, window_size=1024) -> None:
        super().__init__()
        self.ibn_tokenizer1 = IbnNet(1, mid_channels, channels)
        self.ibn_tokenizer2 = IbnNet(1, mid_channels, channels)
        self.pe = PositionEmbeddingSine()
        self.ibn_encoder = IbnNet(channels, mid_channels, channels)
        self.ibn_decoder = IbnNet(1, mid_channels, channels)
        self.variation = variation
        if self.variation:
            length = window_size // (1 << 6)
            self.z_mu = nn.Linear(channels * length, length)
            self.z_log_var = nn.Linear(channels * length, length)
        else:
            self.z_linear = nn.Linear(channels * length, length)
        self.example_encoder = ExampleEncoder(self_attention, channels, n_heads, dropout, mid_channels, use_ins, n_layers)
        self.sample_encoder = SampleEncoder(self_attention, channels, n_heads, dropout, mid_channels, use_ins, n_layers)
        self.decoder = Decoder(self_attention, channels, n_heads, dropout, mid_channels, use_ins, n_layers)
        self.conv_struct = nn.Conv1d(channels, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, examples, samples, gt_apps=None):
        """
        Args:
            examples (N, L): input examples
            samples (N, L): input samples
        """
        examples = self.ibn_tokenizer1(examples[:, None, :])
        samples = self.ibn_tokenizer2(samples[:, None, :])
        # position embeddings
        # examples = examples + self.pe(examples)
        # samples = samples + self.pe(samples)
        # encode
        examples = self.example_encoder(examples)
        samples = self.sample_encoder(samples, examples)
        # z: (N, D, L)
        z = self.ibn_encoder(samples[-1])
        z = z.flatten(start_dim=1)
        if self.variation:
            mu = self.z_mu(z)
            logvar = self.z_log_var(z)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps * std + mu
        else:
            z = self.z_linear(z)
        # decode
        pred_apps = self.ibn_decoder(z[:, None, :])
        pred_apps = self.decoder(pred_apps, reversed(samples))
        pred_apps = self.conv_struct(pred_apps)
        pred_apps = torch.relu(pred_apps).squeeze(1)
        if self.training:
            if self.variation:
                return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0) + ((gt_apps - pred_apps)**2).mean()
            else:
                return ((gt_apps - pred_apps)**2).mean()
        else:
            return pred_apps

