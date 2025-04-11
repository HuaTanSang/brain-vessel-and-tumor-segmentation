import torch 
import torch.nn as nn 

from competitiveconfusionblock import CFB 
from opticdiscgradientadjustmentalgorithm import ODGA
from trumpetattention import TrumpetAttention


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        # Sử dụng CFB làm khối chính
        self.cfb = CFB(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.cfb(x)
        p = self.pool(out)
        return out, p

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        # Sử dụng CFB để kết hợp tính năng từ nhánh skip và đầu ra upsample
        self.cfb = CFB(in_channels, out_channels)  # in_channels sẽ là tổng các kênh sau concat

    def forward(self, x, skip):
        x = self.up(x)
        # Concat theo kênh
        x = torch.cat([x, skip], dim=1)
        x = self.cfb(x)
        return x

class TAOD_CFNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, num_classes=1):
        super(TAOD_CFNet, self).__init__()
        # Tiền xử lý bằng ODGA
        self.odga = ODGA()
        
        # Encoder
        self.enc1 = EncoderBlock(in_channels, base_channels)      # out: base_channels, pooled -> base_channels
        self.enc2 = EncoderBlock(base_channels, base_channels * 2)    # 2x
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4)  # 4x
        self.enc4 = EncoderBlock(base_channels * 4, base_channels * 8)  # 8x

        # Bottleneck CFB
        self.bottleneck = CFB(base_channels * 8, base_channels * 16)

        # Attention module TAM (áp dụng cho bottleneck hoặc skip connections)
        self.tam = TrumpetAttention(base_channels * 16, inter_channels=base_channels * 4)

        # Decoder
        self.dec4 = DecoderBlock(base_channels * 16, base_channels * 8)
        self.dec3 = DecoderBlock(base_channels * 8, base_channels * 4)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2)
        self.dec1 = DecoderBlock(base_channels * 2, base_channels)

        # Lớp cuối: chuyển số kênh về số lớp cần dự đoán (ví dụ: 1 cho nhị phân)
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # Áp dụng ODGA tiền xử lý (nếu ảnh đầu vào grayscale)
        x = self.odga(x)

        # Encoder
        s1, p1 = self.enc1(x)  # s1: skip connection
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)

        # Bottleneck
        b = self.bottleneck(p4)
        # Áp dụng TAM để tăng cường đặc trưng bottleneck
        b = self.tam(b)

        # Decoder
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        out = self.final_conv(d1)
        # Nếu là phân đoạn nhị phân, có thể sử dụng sigmoid ở cuối
        out = torch.sigmoid(out)
        return out