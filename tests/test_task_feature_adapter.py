"""Test Suite 2 — TaskFeatureAdapter Tests"""
import torch
from geofm.models.peft import TaskFeatureAdapter


def test_task_feature_adapter_shapes():
    """Verify all 4 feature levels are processed."""
    features = [
        torch.randn(2, 768, 64, 64),   # F2
        torch.randn(2, 768, 32, 32),   # F5
        torch.randn(2, 768, 16, 16),   # F8
        torch.randn(2, 768, 8, 8),     # F11
    ]
    
    adapter = TaskFeatureAdapter(feature_dims=[768, 768, 768, 768])
    output = adapter(features)
    
    assert len(output) == 4, f"Expected 4 outputs, got {len(output)}"
    print("  [1] Output count: 4 features - PASS")
    
    for i, (inp, out) in enumerate(zip(features, output)):
        assert out.shape == inp.shape, f"F{i+2} shape mismatch: {out.shape} vs {inp.shape}"
    print("  [2] All shapes preserved: PASS")


def test_task_feature_adapter_different_dims():
    """Test with different feature dimensions."""
    features = [
        torch.randn(2, 256, 64, 64),
        torch.randn(2, 512, 32, 32),
        torch.randn(2, 768, 16, 16),
        torch.randn(2, 1024, 8, 8),
    ]
    
    adapter = TaskFeatureAdapter(feature_dims=[256, 512, 768, 1024])
    output = adapter(features)
    
    assert len(output) == 4
    for i, (inp, out) in enumerate(zip(features, output)):
        assert out.shape == inp.shape
    print("  [3] Different dims work: PASS")


if __name__ == "__main__":
    print("\n" + "="*40)
    print("Test Suite 2 — TaskFeatureAdapter")
    print("="*40)
    test_task_feature_adapter_shapes()
    test_task_feature_adapter_different_dims()
    print("TaskFeatureAdapter Tests: PASS ✅\n")