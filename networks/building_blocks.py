# All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

def upsample(x, mode='nearest'):
    """Upsample input tensor by a factor of 2
    mode: select among 'nearest', 'linear', 'bilinear', 'bicubic',
                       'trilinear', 'area', and 'identity'
    """
    if mode == 'identity':
        return x

    return F.interpolate(x, scale_factor=2, mode=mode)

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class TransposeConv3x3(ConvBlock):
    """upsampling layer used by depth for the videos in the wild
    """
    def __init__(self, in_channels, out_channels):
        super(TransposeConv3x3, self).__init__(in_channels, out_channels)

        self.conv = nn.ConvTranspose2d(
                in_channels, out_channels, 3, stride=2, padding=0,
                output_padding=0
                )
        self.nonlin = nn.ReLU(inplace=True)

class WildConvBlock(ConvBlock):
    """Layer to perform a convolution followed by ReLU
    """
    def __init__(self, in_channels, out_channels):
        super(WildConvBlock, self).__init__(in_channels, out_channels)
        self.nonlin = nn.ReLU(inplace=True)
