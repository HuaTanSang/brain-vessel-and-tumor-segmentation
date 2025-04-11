import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class TrumpetAttention(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(TrumpetAttention, self).__init__()
        self.in_channels = in_channels

        self.conv2_H = DoubleConv(in_channels, inter_channels)
        self.conv2dx_H = SingleConv(inter_channels, inter_channels)

        self.conv2_W = DoubleConv(in_channels, inter_channels)
        self.conv2dx_W = SingleConv(inter_channels, inter_channels)

        self.conv2dx_dense = SingleConv(2 * inter_channels, inter_channels)

        # ✅ Thêm bước chuyển Dense về số kênh gốc của X
        self.project_back = nn.Conv2d(inter_channels, in_channels, kernel_size=1)

    def forward(self, X):
        B, C, H, W = X.shape

        # Nhánh H
        X_trans = F.relu(X.permute(0, 1, 3, 2))  # (B, C, W, H)
        H_branch = self.conv2_H(X_trans)
        H_branch = self.conv2dx_H(H_branch)
        H_branch_trans = H_branch.permute(0, 1, 3, 2)  # (B, inter_channels, H, W)

        # Nhánh W
        W_branch = self.conv2_W(X)
        W_branch = self.conv2dx_W(W_branch)

        # Kết hợp
        combined = torch.cat([H_branch_trans, W_branch], dim=1)
        Dense = torch.sigmoid(self.conv2dx_dense(combined))  # (B, inter_channels, H, W)

        # ✅ Chuyển Dense về cùng số kênh với X
        Dense = self.project_back(Dense)  # (B, in_channels, H, W)

        # Tính partial, beta
        partial = Dense.mean(dim=[1, 2, 3], keepdim=True)
        Dense_flat = Dense.view(B, -1)
        max_dense = Dense_flat.max(dim=1)[0].view(B, 1, 1, 1)
        min_dense = Dense_flat.min(dim=1)[0].view(B, 1, 1, 1)
        sum_dense = Dense_flat.sum(dim=1).view(B, 1, 1, 1)
        beta = (max_dense + sum_dense) / (sum_dense - min_dense + 1e-6)

        # Cộng đặc trưng đã tăng cường vào X
        out = torch.where(Dense > partial, X + Dense * beta, X)
        return out
