"""geofm.losses.segmentation_losses

Segmentation loss functions.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def build_segmentation_loss(
    loss_type: str = "cross_entropy",
    ignore_index: int = -100,
) -> nn.Module:
    """Build segmentation loss.

    Args:
        loss_type: Type of loss ('cross_entropy', 'dice', 'focal')
        ignore_index: Index to ignore in loss computation

    Returns:
        Loss function
    """
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    elif loss_type == "dice":
        return DiceLoss()
    elif loss_type == "focal":
        return FocalLoss()
    else:
        return nn.CrossEntropyLoss(ignore_index=ignore_index)


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute dice loss.

        Args:
            logits: Model output (B, C, H, W)
            targets: Target labels (B, H, W)

        Returns:
            Dice loss
        """
        # Get predictions
        preds = logits.argmax(dim=1)

        # Flatten
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        # Compute dice coefficient
        intersection = (preds_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            preds_flat.sum() + targets_flat.sum() + self.smooth
        )

        return 1 - dice


class FocalLoss(nn.Module):
    """Focal loss for segmentation."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Model output (B, C, H, W)
            targets: Target labels (B, H, W)

        Returns:
            Focal loss
        """
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combined CE + Dice loss."""

    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            logits: Model output (B, C, H, W)
            targets: Target labels (B, H, W)

        Returns:
            Combined loss
        """
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice