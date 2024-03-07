import torch
from torch import nn

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
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ffd, dropout) -> None:
        super().__init__()
        self.attention = Attention(d_model, n_heads)
        self.ffd = nn.Sequential(
            nn.Linear(d_model, d_ffd),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffd, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.norm1(self.attention(x,x,x)))
        x = x + self.dropout2(self.norm2(self.ffd(x)))
        return x
    
class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ffd, dropout, n_layer) -> None:
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ffd, dropout) for _ in range(n_layer)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ffd, dropout) -> None:
        super().__init__()
        self.attention1 = Attention(d_model, n_heads)
        self.attention2 = Attention(d_model, n_heads)
        self.ffd = nn.Sequential(
            nn.Linear(d_model, d_ffd),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffd, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, ef):
        x = x + self.dropout1(self.norm1(self.attention1(x,x,x)))
        x = x + self.dropout2(self.norm2(self.attention2(x,ef,ef)))
        x = x + self.dropout3(self.norm3(self.ffd(x)))
        return x
    
class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ffd, dropout, n_layer) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ffd, dropout) for _ in range(n_layer)])

    def forward(self, x, ef):
        for layer in self.layers:
            x = layer(x, ef)
        return x
    
class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, d_ffd, dropout, n_layer) -> None:
        super().__init__()
        self.encoder = Encoder(d_model, n_heads, d_ffd, dropout, n_layer)
        self.decoder = Decoder(d_model, n_heads, d_ffd, dropout, n_layer)
    def forward(self, sample, example):
        ef = self.encoder(sample)
        x = self.decoder(example, ef)
        return x


