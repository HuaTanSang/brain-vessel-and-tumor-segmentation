import torch
import torch.nn as nn

import torch
import torch.nn as nn

class ODGA(nn.Module):
    def __init__(self):
        super(ODGA, self).__init__()

    def forward(self, x):
        """
        x: tensor đầu vào shape (B, 3, H, W) - ảnh RGB (3 kênh giống nhau).
        Output: tensor sau khi điều chỉnh độ sáng vùng đĩa thị.
        """
        B, C, H, W = x.shape
        assert C == 3, "ODGA yêu cầu ảnh có 3 kênh (RGB xám nhân 3 kênh giống nhau)"

        out = torch.zeros_like(x)
        epsilon = 1e-6

        for c in range(C):
            x_c = x[:, c:c+1, :, :]

            x_flat = x_c.view(B, -1)
            max_val = x_flat.max(dim=1, keepdim=True)[0]
            min_val = x_flat.min(dim=1, keepdim=True)[0]

            eqs1 = max_val / 2
            step = (7/8) * (1 - min_val / (max_val + epsilon))
            bound = (3/4) * (1 - min_val / (max_val + epsilon))

            mask_high = (x_c > eqs1.view(B, 1, 1, 1)).float()
            x_light = x_c * (1 - mask_high) + (x_c * step.view(B, 1, 1, 1)) * mask_high

            device = x.device
            i_idx = torch.arange(H, device=device).view(H, 1).expand(H, W)
            j_idx = torch.arange(W, device=device).view(1, W).expand(H, W)
            idx_matrix = (i_idx * W + j_idx).float().unsqueeze(0).unsqueeze(0) + 1e-6

            numerator = (x_light / idx_matrix).view(B, -1).sum(dim=1)
            eqs2 = numerator.view(B, 1, 1, 1)

            mask_critical = (x_light > eqs2).float()
            x_out = x_light * (1 - mask_critical) + (x_light * bound.view(B, 1, 1, 1)) * mask_critical

            out[:, c:c+1, :, :] = x_out

        return out
