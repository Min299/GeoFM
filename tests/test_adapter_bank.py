"""Test Suite 3 — AdapterBank Routing Tests"""
import torch
from geofm.models.peft import AdapterBank, TaskFeatureAdapter


def test_adapter_bank_registration():
    """Verify adapters can be registered."""
    bank = AdapterBank()
    
    flood_adapter = TaskFeatureAdapter(feature_dims=[768, 768, 768, 768])
    burn_adapter = TaskFeatureAdapter(feature_dims=[768, 768, 768, 768])
    
    bank.register_task("flood", flood_adapter)
    bank.register_task("burn", burn_adapter)
    
    assert "flood" in bank.adapters
    assert "burn" in bank.adapters
    print("  [1] Registration: PASS")


def test_adapter_bank_routing():
    """Verify correct adapter is called for each task."""
    bank = AdapterBank()
    
    features = [
        torch.randn(2, 768, 64, 64),
        torch.randn(2, 768, 32, 32),
        torch.randn(2, 768, 16, 16),
        torch.randn(2, 768, 8, 8),
    ]
    
    flood_adapter = TaskFeatureAdapter(feature_dims=[768, 768, 768, 768])
    burn_adapter = TaskFeatureAdapter(feature_dims=[768, 768, 768, 768])
    
    bank.register_task("flood", flood_adapter)
    bank.register_task("burn", burn_adapter)
    
    # Call flood
    out_flood = bank("flood", features)
    assert len(out_flood) == 4
    print("  [2] Flood routing: PASS")
    
    # Call burn
    out_burn = bank("burn", features)
    assert len(out_burn) == 4
    print("  [3] Burn routing: PASS")


def test_adapter_bank_unknown_task():
    """Verify error on unknown task."""
    bank = AdapterBank()
    
    try:
        bank("unknown", [])
        assert False, "Should have raised KeyError"
    except KeyError:
        print("  [4] Unknown task error: PASS")


if __name__ == "__main__":
    print("\n" + "="*40)
    print("Test Suite 3 — AdapterBank Routing")
    print("="*40)
    test_adapter_bank_registration()
    test_adapter_bank_routing()
    test_adapter_bank_unknown_task()
    print("AdapterBank Tests: PASS ✅\n")