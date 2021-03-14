from collections import namedtuple

import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as torchvision


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DoubleConv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=padding),
            #            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=padding),
            #           nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self,
                 up_in_channels,
                 x_in_channels,
                 kernel_size=3,
                 padding=1,
                 dropout=0):
        super(Decoder, self).__init__()

        self.upsample = nn.ConvTranspose2d(up_in_channels,
                                           up_in_channels // 2,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           output_padding=1)

        in_channels = up_in_channels // 2 + x_in_channels
        out_channels = in_channels // 2

        self.layers = DoubleConv(in_channels, out_channels, kernel_size,
                                 padding)

    def forward(self, x, up):
        up = self.upsample(up)
        return self.layers(torch.cat([x, up], dim=1))


ResnetMeta = namedtuple('ResnetMeta', ['func', 'features_scale'])
resnets = {'resnet101': ResnetMeta(torchvision.models.resnet101, 4)}


class ResnetUnet(nn.Module):
    def __init__(self, resnet):

        resnet_meta = resnets[resnet]
        backbone = resnet_meta.func(pretrained=True)

        super(ResnetUnet, self).__init__()

        self.encoder0 = nn.Sequential(backbone.conv1, backbone.bn1,
                                      backbone.relu)
        self.maxpool = backbone.maxpool

        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        channel_factor = resnet_meta.features_scale

        self.middle = DoubleConv(512 * channel_factor, 512 * channel_factor)

        self.decoder0 = Decoder(512 * channel_factor, 256 * channel_factor)
        self.decoder1 = Decoder(256 * channel_factor, 128 * channel_factor)
        self.decoder2 = Decoder(128 * channel_factor, 64 * channel_factor)
        self.decoder3 = Decoder(64 * channel_factor, 64)

        channels = (64 * channel_factor // 2 + 64) // 2

        self.resize = nn.ConvTranspose2d(channels,
                                         16,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         output_padding=1)

        self.final_layer = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1))

        self.final = nn.Sequential()

    def forward(self, x):
        a1 = self.encoder0(x)
        a1_pooled = self.maxpool(a1)
        a2 = self.encoder1(a1_pooled)
        a3 = self.encoder2(a2)
        a4 = self.encoder3(a3)
        a5 = self.encoder4(a4)
        mid = self.middle(a5)

        d1 = self.decoder0(a4, mid)
        d2 = self.decoder1(a3, d1)
        d3 = self.decoder2(a2, d2)
        d4 = self.decoder3(a1, d3)

        resized = self.resize(d4)
        final = self.final_layer(resized)

        return final
