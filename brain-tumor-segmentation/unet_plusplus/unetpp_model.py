import torch.nn as nn
from unetpp_module import *

class UNET_PP(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        # Backbone
        self.down_00 = DoubleConvolution(self.n_channels, 64)             # Output: 64 channels
        self.down_10 = DownScaling(64, 128)                                 # Output: 128 channels, resolution halved
        self.up_10   = UpScaling(128, 64)                                   # Up: 128 -> 64
        self.down_20 = DownScaling(128, 256)                                # Output: 256 channels
        self.up_20   = UpScaling(256, 128)                                  # Up: 256 -> 128
        self.down_30 = DownScaling(256, 512)                                # Output: 512 channels
        self.up_30   = UpScaling(512, 256)                                  # Up: 512 -> 256
        self.down_40 = DownScaling(512, 1024)                               # Output: 1024 channels
        self.up_40   = UpScaling(1024, 512)                                 # Up: 1024 -> 512

        # First Dense Layer
        self.cat_01 = Concat_and_Conv_Layer(64*2, 64)                       # 64+64 = 128 -> 64
        self.cat_11 = Concat_and_Conv_Layer(128*2, 128)                     # 128+128 = 256 -> 128
        self.cat_21 = Concat_and_Conv_Layer(256*2, 256)                     # 256+256 = 512 -> 256

        # Second Dense Layer
        self.cat_02 = Concat_and_Conv_Layer(64*2 + 64, 128)                 # x_00 (64), x_01 (64), up_11(x_11) (64) = 192 -> 128
        self.up_11  = UpScaling(128, 64)                                    # Up: 128 -> 64
        self.cat_12 = Concat_and_Conv_Layer(128*2 + 128, 256)               # x_10 (128), x_11 (128), up_21(x_21) (128) = 384 -> 256
        self.up_21  = UpScaling(256, 128)                                   # Up: 256 -> 128

        # Third Dense Layer
        self.cat_03 = Concat_and_Conv_Layer(320, 512)                       # x_00 (64) + x_01 (64) + x_02 (128) + up_12(x_12) (64) = 320 channels
        self.up_12  = UpScaling(256, 64)                                    # Up: 256 -> 64

        # Last Layer
        self.cat_31 = Concat_and_Conv_Layer(512*2, 512)                     # x_30 (512) + up_40(x_40) (512) = 1024 -> 512
        self.up_31  = UpScaling(512, 256)                                   # Up: 512 -> 256
        self.cat_22 = Concat_and_Conv_Layer(256*3, 256)                     # x_20 (256) + x_21 (256) + up_31(x_31) (256) = 768 -> 256
        self.cat_13 = Concat_and_Conv_Layer(128*3 + 256, 128)               # x_10 (128) + x_11 (128) + x_12 (256) + up_22(x_22) (128) = 640 -> 128
        self.up_22  = UpScaling(256, 128)                                   # Up: 256 -> 128
        self.cat_04 = Concat_and_Conv_Layer(64*3 + 128 + 512, 64)           # x_00 (64) + x_01 (64) + x_02 (128) + x_03 (512) + up_13(x_13) (64) = 832 -> 64
        self.up_13  = UpScaling(128, 64)                                    # Up: 128 -> 64

        # Output
        self.output_conv = OutConvolution(64, self.n_classes)

    def forward(self, x):
        # Backbone forward
        x_00 = self.down_00(x)         # [B, 64, H, W]
        x_10 = self.down_10(x_00)      # [B, 128, H/2, W/2]
        x_20 = self.down_20(x_10)      # [B, 256, H/4, W/4]
        x_30 = self.down_30(x_20)      # [B, 512, H/8, W/8]
        x_40 = self.down_40(x_30)      # [B, 1024, H/16, W/16]

        # First Dense Layer
        x_01 = self.cat_01(x_00, self.up_10(x_10))      # [B, 64, H, W]
        x_11 = self.cat_11(x_10, self.up_20(x_20))        # [B, 128, H/2, W/2]
        x_21 = self.cat_21(x_20, self.up_30(x_30))        # [B, 256, H/4, W/4]

        # Second Dense Layer
        x_02 = self.cat_02(x_00, x_01, self.up_11(x_11))  # [B, 128, H, W]
        x_12 = self.cat_12(x_10, x_11, self.up_21(x_21))  # [B, 256, H/2, W/2]

        # Third Dense Layer
        x_03 = self.cat_03(x_00, x_01, x_02, self.up_12(x_12))  # [B, 512, H, W]

        # Last Layer
        x_31 = self.cat_31(x_30, self.up_40(x_40))              # [B, 512, H/8, W/8]
        x_22 = self.cat_22(x_20, x_21, self.up_31(x_31))         # [B, 256, H/4, W/4]
        x_13 = self.cat_13(x_10, x_11, x_12, self.up_22(x_22))   # [B, 128, H/2, W/2]
        x_04 = self.cat_04(x_00, x_01, x_02, x_03, self.up_13(x_13))  # [B, 64, H, W]

        # Output
        output = self.output_conv(x_04)
        return output