"""
Reproduced model of VAE

written by lily
email: lily231147@gmail.com
"""
import torch
from torch import nn


class VaeEncoderBlock(nn.Module):
    """
    VaeEncoderBlock is used as a encoder block of VAE, just like the paper open-sourced
    """

    def __init__(self, in_channel, kernel_size, stride, res=True, in_=True):
        super(VaeEncoderBlock, self).__init__()
        self.sub_block = nn.Sequential(
            nn.Conv1d(in_channel, 64, kernel_size, stride=stride, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, (1,), stride=stride),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 256, kernel_size, stride=stride, padding=1),
            nn.BatchNorm1d(256)
        )
        self.res, self.in_ = res, in_
        if self.in_:
            self.in_norm = nn.InstanceNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        """ vae encoder block """
        x = x + self.sub_block(x) if self.res else self.sub_block(x)
        x = self.in_norm(x) if self.in_ else x
        return self.relu(x)


class VaeNet(nn.Module):
    """ VaeNet """

    def __init__(self):
        super(VaeNet, self).__init__()

        # encoder
        self.layer1_0 = VaeEncoderBlock(in_channel=2, kernel_size=3, stride=1)
        self.layer1_1 = nn.MaxPool1d(kernel_size=2)
        self.layer2_0 = VaeEncoderBlock(in_channel=256, kernel_size=3, stride=1)
        self.layer2_1 = nn.MaxPool1d(kernel_size=2)
        self.layer3_0 = VaeEncoderBlock(in_channel=256, kernel_size=3, stride=1)
        self.layer3_1 = nn.MaxPool1d(kernel_size=2)
        self.layer4_0 = VaeEncoderBlock(in_channel=256, kernel_size=3, stride=1)
        self.layer4_1 = nn.MaxPool1d(kernel_size=2)
        self.layer5_0 = VaeEncoderBlock(in_channel=256, kernel_size=3, stride=1)
        self.layer5_1 = nn.MaxPool1d(kernel_size=2)
        self.layer6_0 = VaeEncoderBlock(in_channel=256, kernel_size=3, stride=1, in_=False)
        self.layer6_1 = nn.MaxPool1d(kernel_size=2)
        self.layer7_0 = VaeEncoderBlock(in_channel=256, kernel_size=3, stride=1, in_=False)

        # mid
        self.z_mu = nn.Linear(256 * 16, 16)
        self.z_log_var = nn.Linear(256 * 16, 16)

        # decoder
        self.layer8_0 = VaeEncoderBlock(in_channel=1, kernel_size=3, stride=1, in_=False)
        self.layer8_1 = nn.ConvTranspose1d(512, 256, (3,), (2,), (1,), (1,))
        self.layer9_0 = VaeEncoderBlock(in_channel=256, kernel_size=3, stride=1, in_=False)
        self.layer9_1 = nn.ConvTranspose1d(512, 256, (3,), (2,), (1,), (1,))
        self.layer10_0 = VaeEncoderBlock(in_channel=256, kernel_size=3, stride=1, in_=False)
        self.layer10_1 = nn.ConvTranspose1d(512, 256, (3,), (2,), (1,), (1,))
        self.layer11_0 = VaeEncoderBlock(in_channel=256, kernel_size=3, stride=1, in_=False)
        self.layer11_1 = nn.ConvTranspose1d(512, 256, (3,), (2,), (1,), (1,))
        self.layer12_0 = VaeEncoderBlock(in_channel=256, kernel_size=3, stride=1, in_=False)
        self.layer12_1 = nn.ConvTranspose1d(512, 256, (3,), (2,), (1,), (1,))
        self.layer13_0 = VaeEncoderBlock(in_channel=256, kernel_size=3, stride=1, in_=False)
        self.layer13_1 = nn.ConvTranspose1d(512, 256, (3,), (2,), (1,), (1,))
        self.layer14_0 = VaeEncoderBlock(in_channel=256, kernel_size=3, stride=1, in_=False)
        self.layer14_1 = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=(3,), stride=(1,), padding=1)
        self.relu_layer = nn.ReLU()

    @staticmethod
    def __loss(apps, apps_pred, z_mu, z_log_var):
        kl_loss = torch.mean(-0.5 * torch.sum(1 + z_log_var - z_mu ** 2 - z_log_var.exp(), dim=1), dim=0)
        reconstruct_loss = torch.mean((apps - apps_pred)**2)
        return kl_loss + reconstruct_loss

    def forward(self, examples, aggs, apps=None):
        """ vae forward, she shape of aggs is `(bs, 1, L)` """
        # encoder
        x = torch.cat([examples[:, None, :], aggs[:, None, :]], dim=1)
        x10 = self.layer1_0(x)
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
        x70 = self.layer7_0(x61)  # (bs, 256, L//64=8)

        xflatten = x70.flatten(start_dim=1)
        mu = self.z_mu(xflatten)
        logvar = self.z_log_var(xflatten)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu

        # decoder
        z = z.reshape(-1, 1, 16)
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
        x_pre = self.relu_layer(self.layer14_1(mid_cat)).squeeze(1)
        if self.training:
            return self.__loss(apps, x_pre, mu, logvar)
        else:
            return x_pre