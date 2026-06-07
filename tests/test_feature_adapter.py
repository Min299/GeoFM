"""Test Suite 1 — FeatureAdapter Tests"""
import torch
from geofm.models.peft import FeatureAdapter


def test_feature_adapter_shape():
    """Verify output shape matches input shape."""
    x = torch.randn(2, 768, 64, 64)  # (B, C, H, W)
    adapter = FeatureAdapter(dim=768)
    y = adapter(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    print("  [1] Shape preserved: PASS")


def test_feature_adapter_not_identity():
    """Verify adapter actually modifies input."""
    x = torch.randn(2, 768, 64, 64)
    adapter = FeatureAdapter(dim=768)
    y = adapter(x)
    is_same = torch.allclose(x, y, atol=1e-6)
    assert not is_same, "Adapter output is identical to input!"
    print("  [2] Not identity: PASS")


def test_feature_adapter_residual():
    """Verify residual connection works."""
    x = torch.randn(2, 768, 64, 64)
    adapter = FeatureAdapter(dim=768)
    y = adapter(x)
    
    # y = x + adapter(x) - check adapter component exists
    adapter_component = y - x
    assert not torch.allclose(adapter_component, torch.zeros_like(adapter_component)), \
        "No adapter contribution detected"
    print("  [3] Residual flow: PASS")


def test_feature_adapter_different_dims():
    """Test with different feature dimensions."""
    for dim in [256, 512, 768, 1024]:
        x = torch.randn(2, dim, 64, 64)
        adapter = FeatureAdapter(dim=dim, bottleneck_dim=64)
        y = adapter(x)
        assert y.shape == x.shape
    print("  [4] Different dims: PASS")


if __name__ == "__main__":
    print("\n" + "="*40)
    print("Test Suite 1 — FeatureAdapter")
    print("="*40)
    test_feature_adapter_shape()
    test_feature_adapter_not_identity()
    test_feature_adapter_residual()
    test_feature_adapter_different_dims()
    print("FeatureAdapter Tests: PASS ✅\n")