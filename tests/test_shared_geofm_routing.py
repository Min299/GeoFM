"""Test Suite 9 — SharedGeoFM Routing Tests"""
import torch
import torch.nn as nn
from geofm.models.multitask import SharedGeoFM
from geofm.models.peft import AdapterBank, TaskFeatureAdapter
from geofm.models.decoders import DecoderBank, FloodDecoder, BurnDecoder, LULCDecoder


class CallTracker:
    """Track which components were called."""
    def __init__(self):
        self.backbone_called = False
        self.adapter_called = None
        self.decoder_called = None
    
    def reset(self):
        self.backbone_called = False
        self.adapter_called = None
        self.decoder_called = None


class TrackedBackbone(nn.Module):
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker
    
    def forward(self, x):
        self.tracker.backbone_called = True
        return [
            torch.randn(x.shape[0], 768, 64, 64),
            torch.randn(x.shape[0], 768, 32, 32),
            torch.randn(x.shape[0], 768, 16, 16),
            torch.randn(x.shape[0], 768, 8, 8),
        ]


class TrackedAdapterBank(nn.Module):
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker
    
    def forward(self, task_name, features):
        self.tracker.adapter_called = task_name
        return features


class TrackedDecoderBank(nn.Module):
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker
    
    def forward(self, task_name, features):
        self.tracker.decoder_called = task_name
        return torch.randn(features[0].shape[0], 2, 64, 64)


def test_shared_geofm_routing():
    """Test that correct adapter and decoder are called for each task."""
    tracker = CallTracker()
    
    model = SharedGeoFM(
        backbone=TrackedBackbone(tracker),
        adapter_bank=TrackedAdapterBank(tracker),
        decoder_bank=TrackedDecoderBank(tracker),
    )
    
    x = torch.randn(2, 3, 256, 256)
    
    # Test flood
    tracker.reset()
    out = model(x, "flood")
    assert tracker.backbone_called, "Backbone not called"
    assert tracker.adapter_called == "flood", f"Wrong adapter: {tracker.adapter_called}"
    assert tracker.decoder_called == "flood", f"Wrong decoder: {tracker.decoder_called}"
    print("  [1] Flood routing: PASS")
    
    # Test burn
    tracker.reset()
    out = model(x, "burn")
    assert tracker.backbone_called
    assert tracker.adapter_called == "burn"
    assert tracker.decoder_called == "burn"
    print("  [2] Burn routing: PASS")
    
    # Test lulc
    tracker.reset()
    out = model(x, "lulc")
    assert tracker.backbone_called
    assert tracker.adapter_called == "lulc"
    assert tracker.decoder_called == "lulc"
    print("  [3] LULC routing: PASS")


def test_shared_geofm_with_real_components():
    """Test SharedGeoFM with real adapter and decoder banks."""
    adapter_bank = AdapterBank()
    adapter_bank.register_task("flood", TaskFeatureAdapter([768, 768, 768, 768]))
    
    decoder_bank = DecoderBank()
    decoder_bank.register_task("flood", FloodDecoder(in_channels=[768, 768, 768, 768]))
    
    class DummyBackbone(nn.Module):
        def forward(self, x):
            return [
                torch.randn(2, 768, 64, 64),
                torch.randn(2, 768, 32, 32),
                torch.randn(2, 768, 16, 16),
                torch.randn(2, 768, 8, 8),
            ]
    
    model = SharedGeoFM(
        backbone=DummyBackbone(),
        adapter_bank=adapter_bank,
        decoder_bank=decoder_bank,
    )
    
    x = torch.randn(2, 3, 256, 256)
    out = model(x, "flood")
    
    assert out.shape == (2, 2, 64, 64)
    print("  [4] Real components: PASS")


if __name__ == "__main__":
    print("\n" + "="*40)
    print("Test Suite 9 — SharedGeoFM Routing")
    print("="*40)
    test_shared_geofm_routing()
    test_shared_geofm_with_real_components()
    print("SharedGeoFM Routing Tests: PASS ✅\n")