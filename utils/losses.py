# 机构：人工智能研究所
# 人员：东
# 时间：2026/6/9 18:08
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super(SobelEdgeLoss, self).__init__()
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        pred_gx = F.conv2d(probs, self.sobel_x, padding=1)
        pred_gy = F.conv2d(probs, self.sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_gx**2 + pred_gy**2 + 1e-6)

        gt_gx = F.conv2d(targets, self.sobel_x, padding=1)
        gt_gy = F.conv2d(targets, self.sobel_y, padding=1)
        gt_edge = torch.sqrt(gt_gx**2 + gt_gy**2 + 1e-6)

        return F.l1_loss(pred_edge, gt_edge)

class BCEDiceEdgeLoss(nn.Module):
    def __init__(self, bce_weight=0.4, dice_weight=0.5, edge_weight=0.1):
        super(BCEDiceEdgeLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.edge = SobelEdgeLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.edge_weight = edge_weight

    def forward(self, logits, targets):
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        loss_edge = self.edge(logits, targets)
        print(
            loss_bce.item(),
            loss_dice.item(),
            loss_edge.item()
        )
        total_loss = self.bce_weight*loss_bce + self.dice_weight*loss_dice + self.edge_weight*loss_edge
        return total_loss












