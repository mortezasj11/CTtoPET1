""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet_AU(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_AU, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.post_up1 = AfterUp(64, 32)
        self.post_up2 = AfterUp(32, 16)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)      #[512,512,64]    [16,64,512,512]
        x2 = self.down1(x1)   #[256,256,128]
        x3 = self.down2(x2)   #[128,128,256]
        x4 = self.down3(x3)   #[64,64,512]     [16,512,64,64]
        x5 = self.down4(x4)   #[32,32,512]     [16,512,32,32]
        x = self.up1(x5, x4)  #                [16,512,64,64]
        x = self.up2(x, x3)                   #[16,256,128,128]
        x = self.up3(x, x2)                   #[16,128,256,256]
        x = self.up4(x, x1)                   #[16,64,512,512]
        x = self.post_up1(x)   # Added        [16,32,512,512]?
        x = self.post_up2(x) 
        logits = self.outc(x)
        
        return logits

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits