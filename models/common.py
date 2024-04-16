"""
MIT License
Copyright (c) 2024-present lily.

written by lily
email: lily231147@gmail.com
"""

import torch
from torch import nn

class IbnNet(nn.Module):
    def __init__(self, in_channels, out_channels, use_ins=False):
        super().__init__()
        self.use_ins = use_ins
        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = out_channels // 4
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
        if self.use_ins:
            self.in_norm = nn.InstanceNorm1d(256)

    def forward(self, x):
        if self.in_channels == self.out_channels:
            x = x + self.stream(x)
        else:
            x = self.stream(x)
        if self.use_ins:
            x = self.in_norm(x)
        return torch.relu(x)
    
class ExampleEncoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.lstm = nn.LSTM(1, out_channels // 2, batch_first=True, bidirectional=True)

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        return output[:, -1, :]
    
class Transpose(nn.Module):
    def __init__(self): 
        super().__init__()
    def forward(self, x):
        return x.transpose(1, 2)

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


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, dim_feedforward=2048, dropout=0.0):
        super().__init__()
        self.atten = Attention(d_model, nhead, dropout)
        # Implementation of Feedforward model
        self.ff =  nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.Sequential(Transpose(), nn.BatchNorm1d(d_model), Transpose())
        self.norm2 = nn.Sequential(Transpose(), nn.BatchNorm1d(d_model), Transpose())
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x)
        x = x + self.dropout1(self.atten(x, x, x))
        x = x + self.dropout2(self.ff(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer() for _ in range(n_layers)])
        self.n_layers = n_layers

    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x
