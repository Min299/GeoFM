"""geofm.losses.segmentation

Segmentation loss functions for GeoFM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for segmentation.

    Formula: 1 - (2 * |pred ∩ target| + ε) / (|pred| + |target| + ε)
    """

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return 1 - (2 * intersection + self.smooth) / (union + self.smooth)


class CombinedLoss(nn.Module):
    """Combined Dice + CE loss for segmentation."""

    def __init__(self, dice_weight=0.5, ce_weight=0.5, ignore_index=-1):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        dice = self.dice_loss(pred, target_one_hot)
        return self.dice_weight * dice + self.ce_weight * ce


class FocalLoss(nn.Module):
    """Focal loss for imbalanced segmentation."""

    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=-1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(
            pred, target, ignore_index=self.ignore_index, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class BoundaryLoss(nn.Module):
    """Boundary loss for segmentation."""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        return F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)
