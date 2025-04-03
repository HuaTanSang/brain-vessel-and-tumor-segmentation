import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = 2
    def forward(self, x, y):
        b, c, h, w = x.shape
        one_hot_label = F.one_hot(y.squeeze(), c)
        reshaped_x = x.permute(0, 2, 3, 1)
        softmax_x = torch.exp(reshaped_x) / torch.sum(torch.exp(reshaped_x), -1).unsqueeze(-1)
        prob = torch.sum(softmax_x * one_hot_label, -1)
        out = torch.mean(-((1 - prob) ** self.w) * torch.log(prob))
        return out

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1
    def forward(self, x, y):
        B, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = torch.exp(x) / torch.sum(torch.exp(x), -1).view(B, h, w, 1)
        onehot = F.one_hot(y.to(torch.int64), num_classes=c)
        numerator = torch.sum(2 * onehot * x, -1) + self.eps
        denominator = torch.sum(onehot + x, -1) + self.eps
        return 1 - torch.mean((1 - torch.sum(x * onehot, -1)) * (numerator / denominator))

class General_Segment_Loss(nn.Module):
    def __init__(self,scale=10,alpha=10):
        super().__init__()
        self.scale=scale
        self.alpha=alpha
        self.dice=DiceLoss()
        self.focal=FocalLoss()
    def forward(self,x,y):
        x=x.squeeze()
        '''
        x B,10,h,w
        y B,1,h,w
        '''
        return self.scale*(self.dice(x,y)+self.alpha*self.focal(x,y))
    

def compute_iou(pred_mask, true_mask):
    """
    Tính toán IoU (Intersection over Union) cho phân đoạn nhị phân.
    - pred_mask: Mảng đầu ra của model sau khi argmax (shape: B x H x W)
    - true_mask: Ground truth mask (shape: B x H x W)
    """
    pred_mask = (pred_mask == 1)  # Chỉ lấy vùng dự đoán là class 1
    true_mask = (true_mask == 1)  # Chỉ lấy vùng ground truth là class 1

    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()

    iou = intersection / union if union != 0 else 0
    return iou


def compute_dice(pred_mask, true_mask):
    """
    Tính Dice Score dựa trên IoU.
    - pred_mask: Mảng đầu ra của model sau khi argmax (shape: B x H x W)
    - true_mask: Ground truth mask (shape: B x H x W)
    """
    iou = compute_iou(pred_mask, true_mask)
    return 2 * iou / (1 + iou) if iou != 0 else 0


def compute_scores(predicted_masks: list, masks: list) -> dict:
    """
    Tính toán IoU và Dice Score cho tập validation.
    """
    metrics = {
        "iou": compute_iou,
        "dice": compute_dice
    }
    scores = {metric_name: [] for metric_name in metrics}

    for pred_mask, true_mask in zip(predicted_masks, masks):
        for metric_name, scorer in metrics.items():
            scores[metric_name].append(scorer(pred_mask, true_mask)) 

    return {metric_name: np.mean(values) for metric_name, values in scores.items()}

