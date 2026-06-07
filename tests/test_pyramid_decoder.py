"""Test Suite 4 — PyramidDecoder Tests"""
import torch
from geofm.models.decoders import PyramidDecoder


def test_pyramid_decoder_shapes():
    """Verify pyramid decoder processes multi-scale features."""
    F2 = torch.randn(2, 256, 64, 64)
    F5 = torch.randn(2, 256, 32, 32)
    F8 = torch.randn(2, 256, 16, 16)
    F11 = torch.randn(2, 256, 8, 8)
    
    features = [F2, F5, F8, F11]
    
    decoder = PyramidDecoder(in_channels=[256, 256, 256, 256], decoder_channels=256)
    output = decoder(features)
    
    expected_shape = (2, 256, 64, 64)
    assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
    print(f"  [1] Output shape {output.shape}: PASS")


def test_pyramid_decoder_different_scales():
    """Test with varying input scales."""
    F2 = torch.randn(2, 768, 128, 128)
    F5 = torch.randn(2, 768, 64, 64)
    F8 = torch.randn(2, 768, 32, 32)
    F11 = torch.randn(2, 768, 16, 16)
    
    features = [F2, F5, F8, F11]
    
    decoder = PyramidDecoder(in_channels=[768, 768, 768, 768], decoder_channels=256)
    output = decoder(features)
    
    expected_shape = (2, 256, 128, 128)
    assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
    print(f"  [2] Different scales {output.shape}: PASS")


def test_pyramid_decoder_gradients():
    """Verify gradients flow through decoder."""
    F2 = torch.randn(2, 256, 64, 64, requires_grad=True)
    F5 = torch.randn(2, 256, 32, 32, requires_grad=True)
    F8 = torch.randn(2, 256, 16, 16, requires_grad=True)
    F11 = torch.randn(2, 256, 8, 8, requires_grad=True)
    
    features = [F2, F5, F8, F11]
    
    decoder = PyramidDecoder(in_channels=[256, 256, 256, 256], decoder_channels=256)
    output = decoder(features)
    loss = output.sum()
    loss.backward()
    
    for i, f in enumerate(features):
        assert f.grad is not None, f"F{i+2} has no gradients"
    print("  [3] Gradients flow: PASS")


if __name__ == "__main__":
    print("\n" + "="*40)
    print("Test Suite 4 — PyramidDecoder")
    print("="*40)
    test_pyramid_decoder_shapes()
    test_pyramid_decoder_different_scales()
    test_pyramid_decoder_gradients()
    print("PyramidDecoder Tests: PASS ✅\n")