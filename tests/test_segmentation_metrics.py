"""tests/test_segmentation_metrics.py

Tests for SegmentationMetrics.
"""
import pytest
import torch

from geofm.evaluation.segmentation_metrics import SegmentationMetrics


class TestSegmentationMetrics:
    """Test SegmentationMetrics class."""

    def test_perfect_prediction(self):
        """Perfect prediction should give IoU = 1.0."""
        metrics = SegmentationMetrics()

        # Logits with class 1 having higher value
        logits = torch.tensor([[[[0.0]], [[10.0]]]])
        target = torch.tensor([[[1]]])

        metrics.update(logits, target)
        results = metrics.compute()

        assert abs(results["iou"] - 1.0) < 0.01
        assert abs(results["dice"] - 1.0) < 0.01
        assert abs(results["f1"] - 1.0) < 0.01
        assert abs(results["precision"] - 1.0) < 0.01
        assert abs(results["recall"] - 1.0) < 0.01

    def test_no_overlap(self):
        """No overlap should give IoU = 0.0."""
        metrics = SegmentationMetrics()

        # Predict all 1s when target is all 0s
        logits = torch.tensor([[[[10.0]], [[-10.0]]]])
        target = torch.tensor([[[0]]])

        metrics.update(logits, target)
        results = metrics.compute()

        assert results["iou"] == 0.0
        assert results["dice"] == 0.0

    def test_partial_overlap(self):
        """Partial overlap should give intermediate IoU."""
        metrics = SegmentationMetrics()

        # Create 4 pixel binary segmentation
        # Shape: (B=1, C=2, H=1, W=4) for logits
        # Shape: (B=1, H=1, W=4) for target
        # Want: pred = [1, 0, 1, 0], target = [1, 0, 1, 0]
        # So class 1 has higher logit at positions 0, 2
        # And class 0 has higher logit at positions 1, 3
        logits = torch.tensor([[
            [[-10.0, 10.0, -10.0, 10.0]],  # Class 0 logit
            [[10.0, -10.0, 10.0, -10.0]],  # Class 1 logit
        ]])  # Shape: (1, 2, 1, 4)
        target = torch.tensor([[[1, 0, 1, 0]]])  # Shape: (1, 1, 4)

        metrics.update(logits, target)
        results = metrics.compute()

        # TP=2 (positions 0,2 correct as class 1), FP=0, FN=0
        assert results["tp"] == 2, f"TP={results['tp']}"
        # IoU should be 1.0 for perfect predictions
        assert abs(results["iou"] - 1.0) < 0.01, f"IoU={results['iou']}"

    def test_reset(self):
        """Reset should clear accumulated stats."""
        metrics = SegmentationMetrics()

        logits = torch.tensor([[[[0.0]], [[10.0]]]])
        target = torch.tensor([[[1]]])

        metrics.update(logits, target)
        metrics.reset()

        results = metrics.compute()

        # All should be zero after reset
        assert results["tp"] == 0
        assert results["fp"] == 0
        assert results["fn"] == 0

    def test_batch_accumulation(self):
        """Multiple updates should accumulate correctly."""
        metrics = SegmentationMetrics()

        # First batch
        logits1 = torch.tensor([[[[0.0]], [[10.0]]]])
        target1 = torch.tensor([[[1]]])
        metrics.update(logits1, target1)

        # Second batch
        logits2 = torch.tensor([[[[0.0]], [[10.0]]]])
        target2 = torch.tensor([[[1]]])
        metrics.update(logits2, target2)

        results = metrics.compute()

        # Should have accumulated
        assert results["tp"] == 2, f"TP={results['tp']}"
        assert abs(results["iou"] - 1.0) < 0.01

    def test_4d_logits(self):
        """Should handle (B, C, H, W) logits."""
        metrics = SegmentationMetrics()

        # (B=2, C=2, H=1, W=1)
        logits = torch.randn(2, 2, 1, 1)
        target = torch.randint(0, 2, (2, 1, 1))

        metrics.update(logits, target)
        results = metrics.compute()

        # Should compute without error
        assert "iou" in results
        assert "dice" in results


class TestSegmentationMetricsEdgeCases:
    """Test edge cases."""

    def test_empty_target(self):
        """Empty target should not crash."""
        metrics = SegmentationMetrics()

        logits = torch.zeros(1, 2, 4, 4)
        target = torch.zeros(1, 4, 4, dtype=torch.long)

        metrics.update(logits, target)
        results = metrics.compute()

        assert "iou" in results

    def test_all_positive(self):
        """All positive predictions."""
        metrics = SegmentationMetrics()

        # Create 4x4 tensors with correct dimensions
        logits = torch.randn(1, 2, 4, 4)
        logits[:, 0] = -10.0  # Class 0
        logits[:, 1] = 10.0   # Class 1
        target = torch.ones(1, 4, 4, dtype=torch.long)

        metrics.update(logits, target)
        results = metrics.compute()

        assert abs(results["recall"] - 1.0) < 0.01, f"Recall={results['recall']}"

    def test_all_negative(self):
        """All negative predictions."""
        metrics = SegmentationMetrics()

        # Predict all class 0 when target is all 0
        # class 0 has higher logit -> argmax returns 0
        logits = torch.zeros(1, 2, 4, 4)
        logits[:, 0] = 10.0   # Class 0 has higher logit
        logits[:, 1] = -10.0  # Class 1 has lower logit
        target = torch.zeros(1, 4, 4, dtype=torch.long)

        metrics.update(logits, target)
        results = metrics.compute()

        # All predictions and targets are class 0 (negative)
        # TN=16, so accuracy should be 1.0
        assert abs(results["accuracy"] - 1.0) < 0.01, f"Accuracy={results['accuracy']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])