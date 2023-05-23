import torch.nn as nn
from .unet_parts import *

class generator_net(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(generator_net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.dilation = nn.Sequential(
          nn.Conv2d(512, 512, 3, 1, 2, dilation=2),
          nn.BatchNorm2d(512),
          nn.LeakyReLU(0.2),
          nn.Conv2d(512, 512, 3, 1, 4, dilation=4),
          nn.BatchNorm2d(512),
          nn.LeakyReLU(0.2),
          nn.Conv2d(512, 512, 3, 1, 6, dilation=6),
          nn.BatchNorm2d(512),
          nn.LeakyReLU(0.2),
          )
        self.outw = OutConv(64, 3)
        self.outa = OutConv(64, 1)
        self.out_mask = OutConv(64, 1)
        self.sg = nn.Sigmoid()
        self.other = OutConv(64, 64)
        self.post_process_1 = nn.Sequential(
          nn.Conv2d(64+6, 64, 3, 1, 1),
          nn.BatchNorm2d(64),
          nn.LeakyReLU(0.2),
          nn.Conv2d(64, 128, 3, 1, 1),
          )
        self.post_process_2 = nn.Sequential(
          nn.BatchNorm2d(128),
          nn.LeakyReLU(0.2),
          nn.Conv2d(128, 128, 3, 1, 1),
          )
        self.post_process_3 = nn.Sequential(
          nn.BatchNorm2d(128),
          nn.LeakyReLU(0.2),
          nn.Conv2d(128, 128, 3, 1, 1),
          )
        self.post_process_4 = nn.Sequential(
          nn.BatchNorm2d(128),
          nn.LeakyReLU(0.2),
          nn.Conv2d(128, 128, 3, 1, 1),
          )
        self.post_process_5 = nn.Sequential(
          nn.BatchNorm2d(128),
          nn.LeakyReLU(0.2),
          nn.Conv2d(128, 3, 3, 1, 1),
          nn.Sigmoid(),
          )
    def forward(self, x0):
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.dilation(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        w = self.outw(x)
        a = self.outa(x)
        other = self.other(x)
        other = self.sg(other)
        mask = self.out_mask(x)
        mask = self.sg(mask)
        a = self.sg(a)
        w = self.sg(w)
        a = mask * a
        I_watermark = (x0 - a * w) / (1.0 - a + 1e-6)
        I_watermark = torch.clamp(I_watermark, 0, 1)
        xx1 = self.post_process_1(torch.cat([other, I_watermark, x0], 1))
        xx2 = self.post_process_2(xx1)
        xx3 = self.post_process_3(xx1 + xx2)
        xx4 = self.post_process_4(xx2 + xx3)
        I_watermark2 = self.post_process_5(xx4 + xx3)
        I = I_watermark2 * mask + (1.0 - mask) * x0
        # return I, mask, a, w, I_watermark
        return I