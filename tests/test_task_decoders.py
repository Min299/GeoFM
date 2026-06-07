"""Test Suite 5 — Task Decoder Tests"""
import torch
from geofm.models.decoders import FloodDecoder, BurnDecoder, LULCDecoder


def test_flood_decoder():
    """Test flood segmentation decoder."""
    F2 = torch.randn(2, 768, 64, 64)
    F5 = torch.randn(2, 768, 32, 32)
    F8 = torch.randn(2, 768, 16, 16)
    F11 = torch.randn(2, 768, 8, 8)
    
    features = [F2, F5, F8, F11]
    
    decoder = FloodDecoder(in_channels=[768, 768, 768, 768], num_classes=2)
    output = decoder(features)
    
    expected_shape = (2, 2, 64, 64)
    assert output.shape == expected_shape, f"Flood shape: {output.shape} vs {expected_shape}"
    print(f"  [1] FloodDecoder {output.shape}: PASS")


def test_burn_decoder():
    """Test burn scar segmentation decoder."""
    F2 = torch.randn(2, 768, 64, 64)
    F5 = torch.randn(2, 768, 32, 32)
    F8 = torch.randn(2, 768, 16, 16)
    F11 = torch.randn(2, 768, 8, 8)
    
    features = [F2, F5, F8, F11]
    
    decoder = BurnDecoder(in_channels=[768, 768, 768, 768], num_classes=2)
    output = decoder(features)
    
    expected_shape = (2, 2, 64, 64)
    assert output.shape == expected_shape, f"Burn shape: {output.shape} vs {expected_shape}"
    print(f"  [2] BurnDecoder {output.shape}: PASS")


def test_lulc_decoder():
    """Test LULC segmentation decoder."""
    F2 = torch.randn(2, 768, 64, 64)
    F5 = torch.randn(2, 768, 32, 32)
    F8 = torch.randn(2, 768, 16, 16)
    F11 = torch.randn(2, 768, 8, 8)
    
    features = [F2, F5, F8, F11]
    
    decoder = LULCDecoder(in_channels=[768, 768, 768, 768], num_classes=10)
    output = decoder(features)
    
    expected_shape = (2, 10, 64, 64)
    assert output.shape == expected_shape, f"LULC shape: {output.shape} vs {expected_shape}"
    print(f"  [3] LULCDecoder {output.shape}: PASS")


def test_task_decoders_gradients():
    """Verify gradients flow through all decoders."""
    features = [
        torch.randn(2, 768, 64, 64, requires_grad=True),
        torch.randn(2, 768, 32, 32, requires_grad=True),
        torch.randn(2, 768, 16, 16, requires_grad=True),
        torch.randn(2, 768, 8, 8, requires_grad=True),
    ]
    
    # Test Flood
    decoder = FloodDecoder(in_channels=[768, 768, 768, 768])
    output = decoder(features)
    loss = output.sum()
    loss.backward()
    assert features[0].grad is not None
    print("  [4] Flood gradients: PASS")
    
    # Reset gradients
    for f in features:
        f.grad = None
    
    # Test LULC
    decoder = LULCDecoder(in_channels=[768, 768, 768, 768])
    output = decoder(features)
    loss = output.sum()
    loss.backward()
    assert features[0].grad is not None
    print("  [5] LULC gradients: PASS")


if __name__ == "__main__":
    print("\n" + "="*40)
    print("Test Suite 5 — Task Decoders")
    print("="*40)
    test_flood_decoder()
    test_burn_decoder()
    test_lulc_decoder()
    test_task_decoders_gradients()
    print("Task Decoder Tests: PASS ✅\n")