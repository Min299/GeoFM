"""tests/test_segmentation_loss.py

Tests for segmentation loss functions.
"""
import pytest
import torch


class TestSegmentationLoss:
    """Test segmentation losses."""

    def test_cross_entropy_loss(self):
        """CrossEntropyLoss should work."""
        from geofm.losses.segmentation_losses import build_segmentation_loss

        criterion = build_segmentation_loss(loss_type="cross_entropy")

        logits = torch.randn(2, 2, 64, 64)
        target = torch.randint(0, 2, (2, 64, 64))

        loss = criterion(logits, target)

        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_dice_loss(self):
        """Dice loss should work."""
        from geofm.losses.segmentation_losses import DiceLoss

        criterion = DiceLoss()

        logits = torch.randn(2, 2, 64, 64)
        target = torch.randint(0, 2, (2, 64, 64))

        loss = criterion(logits, target)

        assert loss.item() >= 0, "Dice loss should be >= 0"
        assert loss.item() <= 1, "Dice loss should be <= 1"

    def test_focal_loss(self):
        """Focal loss should work."""
        from geofm.losses.segmentation_losses import FocalLoss

        criterion = FocalLoss()

        logits = torch.randn(2, 2, 64, 64)
        target = torch.randint(0, 2, (2, 64, 64))

        loss = criterion(logits, target)

        assert loss.item() > 0, "Focal loss should be positive"
        assert not torch.isnan(loss), "Focal loss should not be NaN"

    def test_combined_loss(self):
        """Combined CE + Dice loss should work."""
        from geofm.losses.segmentation_losses import CombinedLoss

        criterion = CombinedLoss()

        logits = torch.randn(2, 2, 64, 64)
        target = torch.randint(0, 2, (2, 64, 64))

        loss = criterion(logits, target)

        assert loss.item() > 0, "Combined loss should be positive"
        assert not torch.isnan(loss), "Combined loss should not be NaN"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
