import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGGBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(x + self.conv(x))

class CFB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CFB, self).__init__()
        self.vgg_block = VGGBlock(in_channels, out_channels)
        self.res_block = ResidualBlock(out_channels)
        self.omega = 0.5  # hệ số điều chỉnh

    def forward(self, x):
        # Trích xuất đặc trưng từ hai nhánh
        V = self.vgg_block(x)
        R = self.res_block(V)  # đảm bảo cùng shape với V

        # Tổng hợp đặc trưng ban đầu
        C = (V + R) / 2

        # Kích hoạt: M1 và M2
        M1 = self.omega * (torch.sigmoid(V) + F.relu(R))
        M2 = F.relu(R)

        # Tính ngưỡng ∂ từ M2
        B, C_, H, W = M2.shape
        # Dùng trung bình của M2 có trọng số theo vị trí (giả sử sử dụng đơn giản mean)
        partial = M2.view(B, -1).mean(dim=1).view(B, 1, 1, 1)

        # Tạo M3: các phần tử M1 nhỏ hơn partial bị loại bỏ (gán 0)
        M3 = torch.where(M1 > partial, M1, torch.zeros_like(M1))

        # Tính hệ số β từ M1
        max_m1 = M1.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
        min_m1 = M1.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
        sum_m1 = M1.view(B, -1).sum(dim=1).view(B, 1, 1, 1)
        beta = (max_m1 + sum_m1) / (sum_m1 - min_m1 + 1e-6)

        # Kết hợp: nâng cao đặc trưng nếu M1 > partial
        out = torch.where(M1 > partial, beta * M3 * M2, M3 * M2)
        return out