from typing import Optional
import torch
from torch.nn import functional as F
from torch import nn, Tensor
import math

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

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
    def __init__(self, d_model: int = 256, nhead: int = 8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.atten = Attention(d_model, nhead, dropout)
        # Implementation of Feedforward model
        self.ff =  nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.Sequential(Transpose(-2, -1), nn.BatchNorm1d(d_model), Transpose(-2, -1))
        self.norm2 = nn.Sequential(Transpose(-2, -1), nn.BatchNorm1d(d_model), Transpose(-2, -1))
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        x = self.norm1(x)
        x = x + self.dropout1(self.atten(x, x, x))
        x = x + self.dropout2(self.ff(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):

    def __init__(self, n_layers):
        super().__init__()
        self.layers = [TransformerEncoderLayer() for _ in range(n_layers)]
        self.n_layers = n_layers

    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x



