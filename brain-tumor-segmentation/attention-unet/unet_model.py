from torch import nn
from unet_block import *

class UNETModelWithMultiHeadCoAttention(nn.Module):
    def __init__(self, n_channel, n_class, num_heads=8):
        super().__init__()
        self.n_channel = n_channel
        self.n_class = n_class
        self.num_heads = num_heads

        self.first_conv = DoubleConv(n_channel, 64)
        
        self.down1 = DownScaling(64, 128)
        self.down2 = DownScaling(128, 256)
        self.down3 = DownScaling(256, 512)
        self.down4 = DownScaling(512, 1024)

        # Thêm Multi-Head Co-Attention vào các kết nối giữa encoder và decoder
        self.multihead_attention1 = MultiHeadCoAttention(512, num_heads=self.num_heads)
        self.multihead_attention2 = MultiHeadCoAttention(256, num_heads=self.num_heads)
        self.multihead_attention3 = MultiHeadCoAttention(128, num_heads=self.num_heads)
        self.multihead_attention4 = MultiHeadCoAttention(64, num_heads=self.num_heads)

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

        # Decoder với Multi-Head Co-Attention
        x = self.up1(x5, x4)
        x = self.multihead_attention1(x, x4)  # Áp dụng Multi-Head Co-Attention giữa x5 và x4
        x = self.up2(x, x3)
        x = self.multihead_attention2(x, x3)  # Áp dụng Multi-Head Co-Attention giữa x4 và x3
        x = self.up3(x, x2)
        x = self.multihead_attention3(x, x2)  # Áp dụng Multi-Head Co-Attention giữa x3 và x2
        x = self.up4(x, x1)
        x = self.multihead_attention4(x, x1)  # Áp dụng Multi-Head Co-Attention giữa x2 và x1

        return self.final_conv(x)
