"""tests/test_metric_sanity.py

Test metric implementation sanity.
Verify metrics work correctly on known cases.
"""
import pytest
import torch


class TestMetricSanity:
    """Test metric sanity checks."""

    def test_perfect_prediction_accuracy(self):
        """Perfect prediction should give accuracy = 1.0."""
        from geofm.metrics.segmentation_metrics import pixel_accuracy

        preds = torch.tensor([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])  # (B=2, H=2, W=2)
        targets = torch.tensor([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])

        acc = pixel_accuracy(preds, targets)

        assert abs(acc - 1.0) < 1e-6, f"Perfect prediction accuracy should be 1.0, got {acc}"

    def test_worst_prediction_accuracy(self):
        """Completely wrong prediction should give accuracy = 0.0."""
        from geofm.metrics.segmentation_metrics import pixel_accuracy

        preds = torch.tensor([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])
        targets = torch.tensor([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])  # Opposite

        acc = pixel_accuracy(preds, targets)

        assert abs(acc - 0.0) < 1e-6, f"Wrong prediction accuracy should be 0.0, got {acc}"

    def test_perfect_dice(self):
        """Perfect prediction should give Dice = 1.0."""
        from geofm.metrics.segmentation_metrics import dice_coefficient

        preds = torch.tensor([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])
        targets = preds.clone()

        dice = dice_coefficient(preds, targets, num_classes=2)

        assert abs(dice - 1.0) < 1e-6, f"Perfect prediction Dice should be 1.0, got {dice}"

    def test_perfect_iou(self):
        """Perfect prediction should give IoU = 1.0."""
        from geofm.metrics.segmentation_metrics import iou

        preds = torch.tensor([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])
        targets = preds.clone()

        iou_score = iou(preds, targets, num_classes=2)

        assert abs(iou_score - 1.0) < 1e-6, f"Perfect prediction IoU should be 1.0, got {iou_score}"

    def test_random_prediction_metrics(self):
        """Random prediction should give metrics between 0 and 1."""
        from geofm.metrics.segmentation_metrics import (
            pixel_accuracy,
            dice_coefficient,
            iou,
        )

        # Create random predictions and targets
        preds = torch.randint(0, 2, (2, 32, 32))
        targets = torch.randint(0, 2, (2, 32, 32))

        acc = pixel_accuracy(preds, targets)
        dice = dice_coefficient(preds, targets, num_classes=2)
        iou_score = iou(preds, targets, num_classes=2)

        # All metrics should be between 0 and 1
        assert 0 <= acc <= 1, f"Accuracy out of range: {acc}"
        assert 0 <= dice <= 1, f"Dice out of range: {dice}"
        assert 0 <= iou_score <= 1, f"IoU out of range: {iou_score}"

    def test_metric_with_4d_input(self):
        """Metrics should handle 4D logits input."""
        from geofm.metrics.segmentation_metrics import (
            pixel_accuracy,
            dice_coefficient,
            iou,
        )

        # Create 4D logits (B, C, H, W)
        logits = torch.randn(2, 2, 32, 32)
        targets = torch.randint(0, 2, (2, 32, 32))

        # Metrics should handle this by argmax
        acc = pixel_accuracy(logits, targets)
        dice = dice_coefficient(logits, targets, num_classes=2)
        iou_score = iou(logits, targets, num_classes=2)

        assert 0 <= acc <= 1
        assert 0 <= dice <= 1
        assert 0 <= iou_score <= 1

    def test_f1_score_perfect(self):
        """Perfect prediction should give F1 = 1.0."""
        from geofm.metrics.segmentation_metrics import f1_score

        preds = torch.tensor([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])
        targets = preds.clone()

        f1 = f1_score(preds, targets, num_classes=2)

        assert abs(f1 - 1.0) < 1e-6, f"Perfect prediction F1 should be 1.0, got {f1}"

    def test_precision_recall(self):
        """Precision and recall should be between 0 and 1."""
        from geofm.metrics.segmentation_metrics import (
            precision,
            recall,
        )

        preds = torch.randint(0, 2, (2, 32, 32))
        targets = torch.randint(0, 2, (2, 32, 32))

        prec = precision(preds, targets, num_classes=2)
        rec = recall(preds, targets, num_classes=2)

        assert 0 <= prec <= 1, f"Precision out of range: {prec}"
        assert 0 <= rec <= 1, f"Recall out of range: {rec}"

    def test_segmentation_metrics_container(self):
        """SegmentationMetrics container should work."""
        from geofm.metrics.segmentation_metrics import SegmentationMetrics

        metrics = SegmentationMetrics(num_classes=2)

        # Add perfect predictions
        preds = torch.randint(0, 2, (2, 2, 32, 32))
        targets = torch.randint(0, 2, (2, 32, 32))

        # Make some correct predictions
        targets_for_pred = targets.clone()
        metrics.update(preds, targets_for_pred)

        results = metrics.compute()

        assert "accuracy" in results
        assert "dice" in results
        assert "iou" in results
        assert "f1" in results

    def test_metrics_reset(self):
        """Metrics should reset properly."""
        from geofm.metrics.segmentation_metrics import SegmentationMetrics

        metrics = SegmentationMetrics(num_classes=2)

        # Add some data
        preds = torch.randint(0, 2, (2, 2, 32, 32))
        targets = torch.randint(0, 2, (2, 32, 32))
        metrics.update(preds, targets)

        # Reset
        metrics.reset()

        # Should have count 0
        results = metrics.compute()
        assert results == {}, f"Reset metrics should be empty, got {results}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])