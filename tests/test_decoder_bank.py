"""Test Suite 8 — DecoderBank Tests"""
import torch
from geofm.models.decoders import DecoderBank, FloodDecoder, BurnDecoder, LULCDecoder


def test_decoder_bank_registration():
    """Verify decoders can be registered."""
    bank = DecoderBank()
    
    flood_decoder = FloodDecoder(in_channels=[768, 768, 768, 768])
    burn_decoder = BurnDecoder(in_channels=[768, 768, 768, 768])
    lulc_decoder = LULCDecoder(in_channels=[768, 768, 768, 768])
    
    bank.register_task("flood", flood_decoder)
    bank.register_task("burn", burn_decoder)
    bank.register_task("lulc", lulc_decoder)
    
    assert "flood" in bank.decoders
    assert "burn" in bank.decoders
    assert "lulc" in bank.decoders
    print("  [1] Registration: PASS")


def test_decoder_bank_routing():
    """Verify correct decoder is called for each task."""
    bank = DecoderBank()
    
    bank.register_task("flood", FloodDecoder(in_channels=[768, 768, 768, 768]))
    bank.register_task("burn", BurnDecoder(in_channels=[768, 768, 768, 768]))
    bank.register_task("lulc", LULCDecoder(in_channels=[768, 768, 768, 768]))
    
    features = [
        torch.randn(2, 768, 64, 64),
        torch.randn(2, 768, 32, 32),
        torch.randn(2, 768, 16, 16),
        torch.randn(2, 768, 8, 8),
    ]
    
    # Test flood
    out = bank("flood", features)
    assert out.shape == (2, 2, 64, 64), f"Flood shape: {out.shape}"
    print("  [2] Flood routing: PASS")
    
    # Test burn
    out = bank("burn", features)
    assert out.shape == (2, 2, 64, 64), f"Burn shape: {out.shape}"
    print("  [3] Burn routing: PASS")
    
    # Test lulc
    out = bank("lulc", features)
    assert out.shape == (2, 10, 64, 64), f"LULC shape: {out.shape}"
    print("  [4] LULC routing: PASS")


def test_decoder_bank_unknown_task():
    """Verify error on unknown task."""
    bank = DecoderBank()
    
    try:
        bank("unknown", [])
        assert False, "Should have raised KeyError"
    except KeyError:
        print("  [5] Unknown task error: PASS")


if __name__ == "__main__":
    print("\n" + "="*40)
    print("Test Suite 8 — DecoderBank")
    print("="*40)
    test_decoder_bank_registration()
    test_decoder_bank_routing()
    test_decoder_bank_unknown_task()
    print("DecoderBank Tests: PASS ✅\n")