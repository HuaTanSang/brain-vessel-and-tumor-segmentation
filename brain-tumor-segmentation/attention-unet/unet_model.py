from torch import nn
from unet_block import *  # Giả sử bạn đã có các lớp DoubleConv, DownScaling, UpScaling, và OutConv từ mã trước

class UNETModelWithAttentionGate(nn.Module):
    def __init__(self, n_channel, n_class):
        super().__init__()
        self.n_channel = n_channel
        self.n_class = n_class

        self.first_conv = DoubleConv(n_channel, 64)
        
        self.down1 = DownScaling(64, 128)
        self.down2 = DownScaling(128, 256)
        self.down3 = DownScaling(256, 512)
        self.down4 = DownScaling(512, 1024)

        # AttentionGate với tham số phù hợp: (F_g, F_l, F_int)
        self.attention_gate1 = AttentionGate(512, 512, 256)
        self.attention_gate2 = AttentionGate(256, 256, 128)
        self.attention_gate3 = AttentionGate(128, 128, 64)
        self.attention_gate4 = AttentionGate(64, 64, 32)

        self.up1 = UpScaling(1024, 512)
        self.up2 = UpScaling(512, 256)
        self.up3 = UpScaling(256, 128)
        self.up4 = UpScaling(128, 64)

        self.final_conv = OutConv(64, n_class)

    def forward(self, x):
        # Encoder
        x1 = self.first_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder với AttentionGate
        x = self.up1(x5, x4)
        x = self.attention_gate1(x, x4)  # Áp dụng Attention Gate giữa output của up1 và x4
        x = self.up2(x, x3)
        x = self.attention_gate2(x, x3)
        x = self.up3(x, x2)
        x = self.attention_gate3(x, x2)
        x = self.up4(x, x1)
        x = self.attention_gate4(x, x1)

        return self.final_conv(x)
