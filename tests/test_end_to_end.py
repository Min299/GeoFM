"""Test Suite 10 — End-to-End Smoke Test"""
import torch
import torch.nn as nn
from geofm.models.multitask import SharedGeoFM
from geofm.models.peft import TaskFeatureAdapter
from geofm.models.decoders import FloodDecoder


class DummyBackbone(nn.Module):
    """Dummy backbone that outputs multi-scale features."""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        B = x.shape[0]
        return [
            torch.randn(B, 768, 64, 64),   # F2
            torch.randn(B, 768, 32, 32),   # F5
            torch.randn(B, 768, 16, 16),   # F8
            torch.randn(B, 768, 8, 8),     # F11
        ]


class SimpleAdapterBank(nn.Module):
    """Simple pass-through adapter bank for testing."""
    def __init__(self):
        super().__init__()
        self.adapters = nn.ModuleDict()
    
    def register_task(self, task_name, adapter):
        self.adapters[task_name] = adapter
    
    def forward(self, task_name, features):
        return self.adapters[task_name](features)


class SimpleDecoderBank(nn.Module):
    """Simple decoder bank for testing."""
    def __init__(self):
        super().__init__()
        self.decoders = nn.ModuleDict()
    
    def register_task(self, task_name, decoder):
        self.decoders[task_name] = decoder
    
    def forward(self, task_name, features):
        return self.decoders[task_name](features)


def test_end_to_end_flood():
    """Test full forward pass for flood task."""
    print("\n  Building model...")
    
    # Build model
    backbone = DummyBackbone()
    
    adapter_bank = SimpleAdapterBank()
    adapter_bank.register_task("flood", TaskFeatureAdapter([768, 768, 768, 768]))
    
    decoder_bank = SimpleDecoderBank()
    decoder_bank.register_task("flood", FloodDecoder(in_channels=[768, 768, 768, 768]))
    
    model = SharedGeoFM(
        backbone=backbone,
        adapter_bank=adapter_bank,
        decoder_bank=decoder_bank,
    )
    
    # Forward pass
    print("  Running forward pass...")
    image = torch.randn(2, 3, 256, 256)
    output = model(image, "flood")
    
    expected_shape = (2, 2, 64, 64)
    assert output.shape == expected_shape, f"Shape: {output.shape} vs {expected_shape}"
    print(f"  [1] Forward pass: {output.shape} - PASS")


def test_end_to_end_backward():
    """Test backward pass with loss."""
    print("\n  Building model with gradients...")
    
    backbone = DummyBackbone()
    
    adapter_bank = SimpleAdapterBank()
    adapter_bank.register_task("flood", TaskFeatureAdapter([768, 768, 768, 768]))
    
    decoder_bank = SimpleDecoderBank()
    decoder_bank.register_task("flood", FloodDecoder(in_channels=[768, 768, 768, 768]))
    
    model = SharedGeoFM(
        backbone=backbone,
        adapter_bank=adapter_bank,
        decoder_bank=decoder_bank,
    )
    
    # Forward + backward
    print("  Running forward + backward...")
    image = torch.randn(2, 3, 256, 256, requires_grad=True)
    output = model(image, "flood")
    
    # Compute loss
    loss = output.mean()
    
    # Backward
    loss.backward()
    
    # Verify gradients exist on adapter and decoder
    adapter_params = list(adapter_bank.parameters())
    decoder_params = list(decoder_bank.parameters())
    
    assert len(adapter_params) > 0, "No adapter parameters"
    assert len(decoder_params) > 0, "No decoder parameters"
    
    # At least some gradients should exist
    has_grads = any(p.grad is not None for p in adapter_params + decoder_params)
    assert has_grads, "No gradients in adapter/decoder"
    
    print("  [2] Backward pass: PASS")


def test_parameter_counts():
    """Report trainable parameter counts."""
    print("\n  Counting parameters...")
    
    backbone = DummyBackbone()
    
    adapter_bank = SimpleAdapterBank()
    adapter_bank.register_task("flood", TaskFeatureAdapter([768, 768, 768, 768]))
    
    decoder_bank = SimpleDecoderBank()
    decoder_bank.register_task("flood", FloodDecoder(in_channels=[768, 768, 768, 768]))
    
    model = SharedGeoFM(
        backbone=backbone,
        adapter_bank=adapter_bank,
        decoder_bank=decoder_bank,
    )
    
    # Count by component
    adapter_params = sum(p.numel() for p in adapter_bank.parameters())
    decoder_params = sum(p.numel() for p in decoder_bank.parameters())
    backbone_params = sum(p.numel() for p in backbone.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    
    print("\n" + "="*50)
    print("PARAMETER COUNTS (Flood Model)")
    print("="*50)
    print(f"  Backbone (frozen):     {backbone_params:>12,}")
    print(f"  Adapter (trainable):   {adapter_params:>12,}")
    print(f"  Decoder (trainable):   {decoder_params:>12,}")
    print("-"*50)
    print(f"  Total trainable:       {adapter_params + decoder_params:>12,}")
    print(f"  Total parameters:      {total_params:>12,}")
    print("="*50)
    
    # Verify adapter and decoder are trainable
    trainable_adapter = sum(p.numel() for p in adapter_bank.parameters() if p.requires_grad)
    trainable_decoder = sum(p.numel() for p in decoder_bank.parameters() if p.requires_grad)
    
    assert trainable_adapter == adapter_params, "Adapter not trainable"
    assert trainable_decoder == decoder_params, "Decoder not trainable"
    
    print("  [3] Parameter counts: PASS")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Test Suite 10 — End-to-End Smoke Test")
    print("="*50)
    
    test_end_to_end_flood()
    test_end_to_end_backward()
    test_parameter_counts()
    
    print("\n" + "="*50)
    print("END-TO-END SMOKE TEST: ALL PASS ✅")
    print("="*50)