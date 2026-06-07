"""Test Suite 6 & 7 — Head Tests"""
import torch
from geofm.models.heads import SegmentationHead, ClassificationHead, RegressionHead


def test_segmentation_head():
    """Test segmentation head."""
    x = torch.randn(2, 256, 64, 64)
    head = SegmentationHead(in_channels=256, num_classes=2)
    output = head(x)
    
    assert output.shape == (2, 2, 64, 64), f"Shape: {output.shape}"
    print("  [1] SegmentationHead (2 classes): PASS")


def test_segmentation_head_10_classes():
    """Test segmentation head with 10 classes."""
    x = torch.randn(2, 256, 64, 64)
    head = SegmentationHead(in_channels=256, num_classes=10)
    output = head(x)
    
    assert output.shape == (2, 10, 64, 64), f"Shape: {output.shape}"
    print("  [2] SegmentationHead (10 classes): PASS")


def test_classification_head():
    """Test classification head for crop task."""
    x = torch.randn(2, 256, 64, 64)
    head = ClassificationHead(in_channels=256, num_classes=13)
    output = head(x)
    
    assert output.shape == (2, 13), f"Shape: {output.shape}"
    print("  [3] ClassificationHead (13 classes): PASS")


def test_classification_head_different_classes():
    """Test with different class counts."""
    for num_classes in [5, 10, 100]:
        x = torch.randn(2, 256, 64, 64)
        head = ClassificationHead(in_channels=256, num_classes=num_classes)
        output = head(x)
        assert output.shape == (2, num_classes)
    print("  [4] ClassificationHead (various): PASS")


def test_regression_head():
    """Test regression head for NDVI."""
    x = torch.randn(2, 256, 64, 64)
    head = RegressionHead(in_channels=256, out_channels=1)
    output = head(x)
    
    assert output.shape == (2, 1, 64, 64), f"Shape: {output.shape}"
    print("  [5] RegressionHead (1 output): PASS")


def test_regression_head_multi_output():
    """Test regression head with multiple outputs."""
    x = torch.randn(2, 256, 64, 64)
    head = RegressionHead(in_channels=256, out_channels=3)
    output = head(x)
    
    assert output.shape == (2, 3, 64, 64), f"Shape: {output.shape}"
    print("  [6] RegressionHead (3 outputs): PASS")


if __name__ == "__main__":
    print("\n" + "="*40)
    print("Test Suite 6 & 7 — Heads")
    print("="*40)
    test_segmentation_head()
    test_segmentation_head_10_classes()
    test_classification_head()
    test_classification_head_different_classes()
    test_regression_head()
    test_regression_head_multi_output()
    print("Head Tests: PASS ✅\n")