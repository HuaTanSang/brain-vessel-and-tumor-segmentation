import torch
import torch.nn as nn
import torch.nn.functional as F

#------------------ Các module con ------------------

class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.Mish()
        )

    def forward(self, x):
        return self.double_conv(x)


class OutConvolution(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DownScaling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.downscaling = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvolution(in_channel, out_channel)
        )

    def forward(self, x):
        return self.downscaling(x)


class UpScaling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Mish()
        )
    def forward(self, x):
        return self.up(x)


class Concat_and_Conv_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, dim=1):
        super().__init__()
        self.dim = dim
        self.conv = DoubleConvolution(in_channels, out_channels)

    def forward(self, *inputs):
        conv_input = torch.cat(inputs, dim=self.dim)
        output = self.conv(conv_input)
        return output