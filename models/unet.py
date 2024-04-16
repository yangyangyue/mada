
import torch
import torch.nn as nn

class conv_block_seq_res_fixe_3(nn.Module):
    def __init__(self, inchannel, outchannel,kernel_size, strides, bn=True, In=True, ResCon=True):
        super(conv_block_seq_res_fixe_3, self).__init__()
        self.a = nn.Sequential()
        self.n = 3

        if bn:
            for i in range(1, self.n + 1):
                conv = nn.Sequential(nn.Conv1d(inchannel, outchannel, kernel_size, strides, padding=1),
                                     nn.BatchNorm1d(outchannel),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                inchannel = outchannel

        else:
            for i in range(1, self.n + 1):
                conv = nn.Sequential(nn.Conv1d(inchannel, outchannel, kernel_size, strides, padding=1),
                                     nn.LeakyReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                inchannel = outchannel

        if In:
            self.In_layer=nn.InstanceNorm1d(outchannel)
        self.relu_layer=nn.ReLU()
    def forward(self,input):
        x = input
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        # if(self.Res):
        # x=x+input
        # if self.In:
        #     x=self.In_layer(x)
        return x  # [64, 256, 128]
    
class block_3(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, scale_factor, pool, upsample, pool_size=2):
        super(block_3, self).__init__()
        self.a = nn.Sequential()

        if pool:
            self.a.add_module('wakaka0', nn.MaxPool1d(kernel_size=pool_size))
        if upsample:
            self.a.add_module('wakaka1', nn.Upsample(scale_factor=scale_factor, mode='linear'))
        self.a.add_module('wakaka2', nn.Conv1d(inchannel, outchannel, kernel_size, padding=1))
        self.a.add_module('wakaka3', nn.BatchNorm1d(outchannel))
        self.relu_layer = nn.ReLU()

    def forward(self, x):
        x = self.a(x)  # [64, 256, 128]
        return self.relu_layer(x)  # [64, 256, 128]


class VAE_method_3_(nn.Module):  # 相比VAE_method_3网络层数降低
    def __init__(self):
        super(VAE_method_3_, self).__init__()

        self.outchannel = 64
        self.catchannel = 64 * 5

        # encoder
        self.layer1_0 = conv_block_seq_res_fixe_3(inchannel=1, outchannel=64, kernel_size=3, strides=1, ResCon=False)
        self.layer1_1 = nn.MaxPool1d(kernel_size=2)
        self.layer2_0 = conv_block_seq_res_fixe_3(inchannel=64, outchannel=128, kernel_size=3, strides=1, ResCon=False)
        self.layer2_1 = nn.MaxPool1d(kernel_size=2)
        self.layer3_0 = conv_block_seq_res_fixe_3(inchannel=128, outchannel=256, kernel_size=3, strides=1, ResCon=False)
        self.layer3_1 = nn.MaxPool1d(kernel_size=2)
        self.layer4_0 = conv_block_seq_res_fixe_3(inchannel=256, outchannel=512, kernel_size=3, strides=1, ResCon=False)
        # self.layer4_0 = nn.LSTM(input_size=256, hidden_size=256, bias=True, batch_first=True, bidirectional=True)
        self.layer4_1 = nn.MaxPool1d(kernel_size=2)
        #self.layer5_0 = conv_block_seq_res_fixe_3(inchannel=512, outchannel=1024, kernel_size=3, strides=1, ResCon=False)

        self.layer5_0 = nn.LSTM(input_size=512, hidden_size=512, bias=True, batch_first=True, bidirectional=True)
        self.tanh = nn.ReLU()
        # mid
        self.z_mu = nn.Linear(4096, 16)
        self.z_log_var = nn.Linear(256 * 16, 16)

        # decoder

        '''stage 4d'''
        self.h1_PT_hd4 = block_3(inchannel=64, outchannel=self.outchannel, kernel_size=3, scale_factor=2, pool=True,
                                upsample=False, pool_size=8)
        self.h2_PT_hd4 = block_3(inchannel=128, outchannel=self.outchannel, kernel_size=3, scale_factor=2, pool=True,
                                upsample=False, pool_size=4)
        self.h3_PT_hd4 = block_3(inchannel=256, outchannel=self.outchannel, kernel_size=3, scale_factor=2, pool=True,
                                upsample=False, pool_size=2)
        self.h4_PT_hd4 = block_3(inchannel=512, outchannel=self.outchannel, kernel_size=3, scale_factor=2, pool=False,
                                upsample=False)
        self.h5_PT_hd4 = block_3(inchannel=1024, outchannel=self.outchannel, kernel_size=3, scale_factor=2, pool=False,
                                upsample=True)
        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = block_3(inchannel=self.outchannel * 5, outchannel=self.outchannel * 5, kernel_size=3,
                               scale_factor=2, pool=False, upsample=False)

        '''stage 3d'''
        self.h1_PT_hd3 = block_3(inchannel=64, outchannel=self.outchannel, kernel_size=3, scale_factor=2, pool=True,
                                upsample=False,
                                pool_size=4)
        self.h2_PT_hd3 = block_3(inchannel=128, outchannel=self.outchannel, kernel_size=3, scale_factor=2, pool=True,
                                upsample=False,
                                pool_size=2)
        self.h3_PT_hd3 = block_3(inchannel=256, outchannel=self.outchannel, kernel_size=3, scale_factor=2, pool=False,
                                upsample=False)
        self.h4_PT_hd3 = block_3(inchannel=self.outchannel * 5, outchannel=self.outchannel, kernel_size=3,
                                scale_factor=2, pool=False, upsample=True)
        self.h5_PT_hd3 = block_3(inchannel=1024, outchannel=self.outchannel, kernel_size=3, scale_factor=4, pool=False,
                                upsample=True)
        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4...)
        self.conv3d_1 = block_3(inchannel=self.outchannel * 5, outchannel=self.outchannel * 5, kernel_size=3,
                               scale_factor=2, pool=False,
                               upsample=False)

        '''stage 2d'''
        self.h1_PT_hd2 = block_3(inchannel=64, outchannel=self.outchannel, kernel_size=3, scale_factor=2, pool=True,
                                upsample=False,
                                pool_size=2)
        self.h2_PT_hd2 = block_3(inchannel=128, outchannel=self.outchannel, kernel_size=3, scale_factor=2, pool=False,
                                upsample=False,
                                pool_size=4)
        self.h3_PT_hd2 = block_3(inchannel=self.outchannel * 5, outchannel=self.outchannel, kernel_size=3,
                                scale_factor=2, pool=False, upsample=True)
        self.h4_PT_hd2 = block_3(inchannel=self.outchannel * 5, outchannel=self.outchannel, kernel_size=3,
                                scale_factor=4, pool=False, upsample=True)
        self.h5_PT_hd2 = block_3(inchannel=1024, outchannel=self.outchannel, kernel_size=3, scale_factor=8, pool=False,
                                upsample=True)
        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv2d_1 = block_3(inchannel=self.outchannel * 5, outchannel=self.outchannel * 5, kernel_size=3,
                               scale_factor=2, pool=False,
                               upsample=False)

        '''stage 1d'''
        self.h1_PT_hd1 = block_3(inchannel=64, outchannel=self.outchannel, kernel_size=3, scale_factor=2, pool=False,
                                upsample=False,
                                pool_size=4)
        self.h2_PT_hd1 = block_3(inchannel=self.outchannel * 5, outchannel=self.outchannel, kernel_size=3,
                                scale_factor=2, pool=False, upsample=True,
                                pool_size=4)
        self.h3_PT_hd1 = block_3(inchannel=self.outchannel * 5, outchannel=self.outchannel, kernel_size=3,
                                scale_factor=4, pool=False, upsample=True,
                                pool_size=2)
        self.h4_PT_hd1 = block_3(inchannel=self.outchannel * 5, outchannel=self.outchannel, kernel_size=3,
                                scale_factor=8, pool=False,
                                upsample=True)
        self.h5_PT_hd1 = block_3(inchannel=1024, outchannel=self.outchannel, kernel_size=3, scale_factor=16, pool=False,
                                upsample=True)
        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv1d_1 = block_3(inchannel=self.outchannel * 5, outchannel=self.outchannel * 5, kernel_size=3,
                               scale_factor=2, pool=False,
                               upsample=False)


        self.last_conv = nn.Conv1d(in_channels=self.outchannel * 5, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.relu_layer = nn.ReLU()

    def forward(self, _, x, apps=None):
        # encorder
        x= x[:, None, :]
        x10 = self.layer1_0(x)
        x11 = self.layer1_1(x10)
        x20 = self.layer2_0(x11)
        x21 = self.layer2_1(x20)
        x30 = self.layer3_0(x21)
        x31 = self.layer3_1(x30)  # [32,256,128]
        x40 = self.layer4_0(x31)
        x41 = self.layer4_1(x40)  # [32, 512, 64]
        # print(x41.shape)
        # exit()
        x50, (h0_, c0_) = self.layer5_0(x41.permute(0, 2, 1))  # [32, 256, 16]

        x50 = self.tanh(x50.permute(0, 2, 1))

        xflatten = x50.flatten(start_dim=1)

        # 中间那一堆
        # mu=self.z_mu(xflatten)  # 32, 16

        # logvar=self.z_log_var(xflatten)  # 32, 16

        z = x50
        # decoder包括跳跃连接
        # z=z.reshape(-1,1,16)   # 32, 1, 16

        h1_PT_hd4 = self.h1_PT_hd4(x10)
        h2_PT_hd4 = self.h2_PT_hd4(x20)
        h3_PT_hd4 = self.h3_PT_hd4(x30)
        h4_PT_hd4 = self.h4_PT_hd4(x40)
        h5_PT_hd4 = self.h5_PT_hd4(z)

        hd4 = self.conv4d_1(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_PT_hd4, h5_PT_hd4), 1))

        h1_PT_hd3 = self.h1_PT_hd3(x10)
        h2_PT_hd3 = self.h2_PT_hd3(x20)
        h3_PT_hd3 = self.h3_PT_hd3(x30)
        h4_PT_hd3 = self.h4_PT_hd3(hd4)
        h5_PT_hd3 = self.h5_PT_hd3(z)

        hd3 = self.conv3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_PT_hd3, h4_PT_hd3, h5_PT_hd3), 1))

        h1_PT_hd2 = self.h1_PT_hd2(x10)
        h2_PT_hd2 = self.h2_PT_hd2(x20)
        h3_PT_hd2 = self.h3_PT_hd2(hd3)
        h4_PT_hd2 = self.h4_PT_hd2(hd4)
        h5_PT_hd2 = self.h5_PT_hd2(z)

        hd2 = self.conv2d_1(torch.cat((h1_PT_hd2, h2_PT_hd2, h3_PT_hd2, h4_PT_hd2, h5_PT_hd2), 1))

        h1_PT_hd1 = self.h1_PT_hd1(x10)
        h2_PT_hd1 = self.h2_PT_hd1(hd2)
        h3_PT_hd1 = self.h3_PT_hd1(hd3)
        h4_PT_hd1 = self.h4_PT_hd1(hd4)
        h5_PT_hd1 = self.h5_PT_hd1(z)

        hd1 = self.conv1d_1(torch.cat((h1_PT_hd1, h2_PT_hd1, h3_PT_hd1, h4_PT_hd1, h5_PT_hd1), 1))

        out = self.last_conv(hd1)
        x_pre = self.relu_layer(out).squeeze(1)
        if self.training:
            return ((apps - x_pre)**2).mean()
        else:
            return x_pre