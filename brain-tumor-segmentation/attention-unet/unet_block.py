from torch import nn 
from torch.nn import functional as F 
import torch 

class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel=None):
        super().__init__()
        if not mid_channel:
            mid_channel = out_channel

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DownScaling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.downscaling = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channel, out_channel)
        )

    def forward(self, x):
        return self.downscaling(x)

class UpScaling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.upscaling = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channel, out_channel)

    def forward(self, x1, x2): # x1 from ConvTransposed, x2 from Encoder
        x1 = self.upscaling(x1)

        delta_height = x2.size()[2] - x1.size()[2]
        delta_width = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [delta_width // 2, delta_width - delta_width // 2,
                        delta_height // 2, delta_height - delta_height // 2])

        x = torch.cat([x2, x1], dim=1)

        return self.double_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

class MultiHeadCoAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(MultiHeadCoAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Các phép biến đổi cho mỗi đầu attention
        self.fc_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x1, x2):
        batch_size, C, H, W = x1.shape

        # Tính toán Query, Key, Value cho mỗi đầu
        Q = self.query(x1).view(batch_size, self.num_heads, self.head_dim, H * W)  # (B, num_heads, head_dim, H*W)
        K = self.key(x2).view(batch_size, self.num_heads, self.head_dim, H * W)    # (B, num_heads, head_dim, H*W)
        V = self.value(x2).view(batch_size, self.num_heads, C // self.num_heads, H * W)  # (B, num_heads, head_dim, H*W)

        # Tính toán attention cho mỗi đầu
        attention = torch.einsum("bhnq,bhkw->bhnqw", Q, K)  # (B, num_heads, H*W, H*W)
        attention = F.softmax(attention, dim=-1)

        # Áp dụng attention vào giá trị (Value)
        out = torch.einsum("bhnqw,bhkw->bhnq", attention, V)  # (B, num_heads, H*W, head_dim)
        out = out.view(batch_size, C, H, W)  # Đưa về kích thước ban đầu

        return self.fc_out(out) + x1  # Residual connection
