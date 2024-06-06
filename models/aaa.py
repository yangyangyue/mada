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


class VaeNet(nn.Module):
    def __init__(self, channels=256, window_size=1024):
        super().__init__()

        # encoder
        self.layer1_0 = IbnNet(1, channels, use_ins=True)
        self.layer1_1 = nn.MaxPool1d(kernel_size=2)
        self.layer2_0 = IbnNet(channels, channels, use_ins=True)
        self.layer2_1 = nn.MaxPool1d(kernel_size=2)
        self.layer3_0 = IbnNet(channels, channels,use_ins=True)
        self.layer3_1 = nn.MaxPool1d(kernel_size=2)
        self.layer4_0 = IbnNet(channels, channels, use_ins=True)
        self.layer4_1 = nn.MaxPool1d(kernel_size=2)
        self.layer5_0 = IbnNet(channels, channels,use_ins=True)
        self.layer5_1 = nn.MaxPool1d(kernel_size=2)
        self.layer6_0 = IbnNet(channels, channels)
        self.layer6_1 = nn.MaxPool1d(kernel_size=2)
        self.layer7_0 = IbnNet(channels, channels)

        encoder_layer = nn.TransformerEncoderLayer(d_model=channels, nhead=1, dim_feedforward=512, dropout=0.1, batch_first=True)
        encoder_norm = nn.LayerNorm(channels)
        self. atten =  nn.TransformerEncoder(encoder_layer, 2, encoder_norm)


        # decoder
        self.layer8_0 = IbnNet(channels, channels, use_ins=False)
        self.layer8_1 = nn.ConvTranspose1d(2 * channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.layer9_0 = IbnNet(channels, channels)
        self.layer9_1 = nn.ConvTranspose1d(2 * channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.layer10_0 = IbnNet(channels, channels)
        self.layer10_1 = nn.ConvTranspose1d(2 * channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.layer11_0 = IbnNet(channels, channels)
        self.layer11_1 = nn.ConvTranspose1d(2 * channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.layer12_0 = IbnNet(channels, channels)
        self.layer12_1 = nn.ConvTranspose1d(2 * channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.layer13_0 = IbnNet(channels, channels)
        self.layer13_1 = nn.ConvTranspose1d(2 * channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.layer14_0 = IbnNet(channels, channels)
        self.layer14_1 = nn.Conv1d(2 * channels, 1, kernel_size=3, stride=1, padding=1)


    def forward(self, aggs, _, apps=None):
        # encoder
        x10 = self.layer1_0(aggs[:, None, :])
        x11 = self.layer1_1(x10)
        x20 = self.layer2_0(x11)
        x21 = self.layer2_1(x20)
        x30 = self.layer3_0(x21)
        x31 = self.layer3_1(x30)
        x40 = self.layer4_0(x31)
        x41 = self.layer4_1(x40)
        x50 = self.layer5_0(x41)
        x51 = self.layer5_1(x50)
        x60 = self.layer6_0(x51)
        x61 = self.layer6_1(x60)
        x70 = self.layer7_0(x61)  # (bs, 256, L//64=16)

        z = self.atten(x70.permute(0, 2,1 )).permute(0,2,1)  # (bs, 256, L//64=16)

        # decoder
        x80 = self.layer8_0(z)
        mid_cat = torch.cat([x80, x70], dim=1)
        x81 = self.layer8_1(mid_cat)
        x90 = self.layer9_0(x81)
        mid_cat = torch.cat([x90, x60], dim=1)
        x91 = self.layer9_1(mid_cat)
        x100 = self.layer10_0(x91)
        mid_cat = torch.cat([x100, x50], dim=1)
        x101 = self.layer10_1(mid_cat)
        x110 = self.layer11_0(x101)
        mid_cat = torch.cat([x110, x40], dim=1)
        x111 = self.layer11_1(mid_cat)
        x120 = self.layer12_0(x111)
        mid_cat = torch.cat([x120, x30], dim=1)
        x121 = self.layer12_1(mid_cat)
        x130 = self.layer13_0(x121)
        mid_cat = torch.cat([x130, x20], dim=1)
        x131 = self.layer13_1(mid_cat)
        x140 = self.layer14_0(x131)
        mid_cat = torch.cat([x140, x10], dim=1)
        x_pre = torch.relu(self.layer14_1(mid_cat)).squeeze(1)
        if self.training:
            return torch.mean((apps - x_pre)**2)
        else:
            return x_pre