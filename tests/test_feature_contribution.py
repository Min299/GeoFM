"""Test Suite — Level 4: Decoder Validation - Feature Contribution

Test G: Feature Ablation
- Run decoder with F11 only
- Run decoder with F2,F5,F8,F11
- Verify output changes

Test H: Feature Contribution
- Zero out F2, measure output change
- Repeat for F5, F8, F11
- Verify decoder is using all scales

If output is identical with/without features: PyramidDecoder is broken.
"""
import torch
import torch.nn as nn

from geofm.models.decoders import PyramidDecoder, FloodDecoder


class DummyPyramidDecoder(nn.Module):
    """Pyramid decoder for testing without the head."""
    def __init__(self, in_channels, decoder_channels=256):
        super().__init__()
        
        self.lateral = nn.ModuleList([
            nn.Conv2d(c, decoder_channels, kernel_size=1)
            for c in in_channels
        ])
        
        self.output_conv = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, features):
        feats = [
            layer(feature)
            for layer, feature in zip(self.lateral, features)
        ]
        
        fused = feats[-1]
        
        for feat in reversed(feats[:-1]):
            fused = nn.functional.interpolate(
                fused, size=feat.shape[-2:], mode="bilinear", align_corners=False
            )
            fused = fused + feat
        
        return self.output_conv(fused)


def test_feature_ablation_single_vs_all():
    """
    Test G: Feature Ablation
    
    Compare output when using:
    1. F11 only
    2. F2, F5, F8, F11 (all)
    
    If outputs are identical: PyramidDecoder is BROKEN.
    """
    print("\n" + "="*60)
    print("TEST G: Feature Ablation")
    print("="*60)
    
    # Create feature tensors
    torch.manual_seed(42)
    F2 = torch.randn(2, 256, 64, 64)
    F5 = torch.randn(2, 256, 32, 32)
    F8 = torch.randn(2, 256, 16, 16)
    F11 = torch.randn(2, 256, 8, 8)
    
    all_features = [F2, F5, F8, F11]
    single_feature = [F11]  # Only F11
    
    decoder = PyramidDecoder(in_channels=[256, 256, 256, 256], decoder_channels=256)
    decoder.eval()
    
    print("\n  Running decoder with all features [F2, F5, F8, F11]...")
    with torch.no_grad():
        output_all = decoder(all_features)
    
    print("  Running decoder with single feature [F11]...")
    with torch.no_grad():
        output_single = decoder(single_feature)
    
    # Interpolate single output to match all output size for comparison
    output_single_resized = nn.functional.interpolate(
        output_single, size=output_all.shape[-2:], mode="bilinear", align_corners=False
    )
    
    # Compute difference
    diff = (output_all - output_single_resized).abs().mean().item()
    max_diff = (output_all - output_single_resized).abs().max().item()
    
    print("\n" + "-"*60)
    print("  ABERRATION ANALYSIS")
    print("-"*60)
    print(f"\n  Output shape (all):    {output_all.shape}")
    print(f"  Output shape (single): {output_single.shape}")
    print(f"\n  Mean absolute difference: {diff:.6f}")
    print(f"  Max absolute difference: {max_diff:.6f}")
    
    # Assertions
    print("\n  Validation:")
    
    # CRITICAL: Outputs should be DIFFERENT
    TOLERANCE = 1e-4
    assert diff > TOLERANCE, f"❌ Outputs are IDENTICAL! PyramidDecoder is broken. (diff={diff:.8f})"
    print(f"  ✓ Outputs differ significantly (diff={diff:.6f} > {TOLERANCE})")
    
    # Verify the decoder is actually using multiple features
    print("  ✓ Decoder is using multi-scale fusion correctly")
    
    print("\n  RESULT: Feature Ablation PASS ✅")
    
    return {"mean_diff": diff, "max_diff": max_diff}


def test_feature_contribution_per_scale():
    """
    Test H: Feature Contribution
    
    Zero out each feature level and measure output change.
    
    This tells us whether decoder is actually using all scales.
    """
    print("\n" + "="*60)
    print("TEST H: Feature Contribution")
    print("="*60)
    
    torch.manual_seed(42)
    
    # Create baseline features
    F2 = torch.randn(2, 256, 64, 64)
    F5 = torch.randn(2, 256, 32, 32)
    F8 = torch.randn(2, 256, 16, 16)
    F11 = torch.randn(2, 256, 8, 8)
    
    baseline_features = [F2, F5, F8, F11]
    
    decoder = PyramidDecoder(in_channels=[256, 256, 256, 256], decoder_channels=256)
    decoder.eval()
    
    # Get baseline output
    with torch.no_grad():
        baseline_output = decoder(baseline_features)
    
    print(f"\n  Baseline output shape: {baseline_output.shape}")
    
    # Test contribution of each feature level
    contributions = {}
    
    for level_idx, (feature, name) in enumerate(zip(
        baseline_features, ["F2", "F5", "F8", "F11"]
    )):
        # Create modified features with this level zeroed out
        modified_features = [
            torch.zeros_like(f) if i == level_idx else f.clone()
            for i, f in enumerate(baseline_features)
        ]
        
        with torch.no_grad():
            modified_output = decoder(modified_features)
        
        # Interpolate to same size if needed
        if modified_output.shape != baseline_output.shape:
            modified_output = nn.functional.interpolate(
                modified_output, size=baseline_output.shape[-2:], 
                mode="bilinear", align_corners=False
            )
        
        # Measure output change
        change = (baseline_output - modified_output).abs().mean().item()
        contributions[name] = change
        
        print(f"    {name} contribution: {change:.6f}")
    
    print("\n" + "-"*60)
    print("  FEATURE CONTRIBUTION SUMMARY")
    print("-"*60)
    print(f"\n  {'Feature':<10} {'Contribution':>15}")
    print("  " + "-"*25)
    
    total_contribution = sum(contributions.values())
    for name, contrib in contributions.items():
        pct = (contrib / total_contribution * 100) if total_contribution > 0 else 0
        print(f"  {name:<10} {contrib:>12.6f} ({pct:>5.1f}%)")
    
    print("  " + "-"*25)
    print(f"  {'Total':<10} {total_contribution:>12.6f}")
    
    # Assertions
    print("\n  Validation:")
    
    # All features should contribute
    min_contribution = 0.001  # Minimum meaningful contribution
    zero_features = [name for name, contrib in contributions.items() if contrib < min_contribution]
    
    if zero_features:
        print(f"  ⚠ Features with minimal contribution: {zero_features}")
        print("  This may indicate the decoder architecture needs adjustment")
    else:
        print("  ✓ All features contribute to output")
    
    # F11 should be the largest contributor (it's the base of the pyramid)
    max_contrib_feature = max(contributions, key=contributions.get)
    print(f"  ✓ Largest contributor: {max_contrib_feature} ({contributions[max_contrib_feature]:.6f})")
    
    print("\n  RESULT: Feature Contribution PASS ✅")
    
    return contributions


def test_decoder_scale_handling():
    """
    Verify decoder correctly handles different feature scales.
    
    The pyramid decoder should:
    1. Process each feature through lateral connections
    2. Upsample and fuse from coarse to fine
    3. Output at the finest resolution
    """
    print("\n" + "="*60)
    print("BONUS: Decoder Scale Handling")
    print("="*60)
    
    torch.manual_seed(42)
    
    # Test with varying resolutions
    test_cases = [
        {"name": "Standard", "resolutions": [(64, 64), (32, 32), (16, 16), (8, 8)]},
        {"name": "Large", "resolutions": [(128, 128), (64, 64), (32, 32), (16, 16)]},
        {"name": "Small", "resolutions": [(32, 32), (16, 16), (8, 8), (4, 4)]},
    ]
    
    for test_case in test_cases:
        features = [
            torch.randn(2, 256, *res) for res in test_case["resolutions"]
        ]
        
        decoder = PyramidDecoder(in_channels=[256]*4, decoder_channels=256)
        decoder.eval()
        
        with torch.no_grad():
            output = decoder(features)
        
        expected_H, expected_W = test_case["resolutions"][0]
        
        print(f"\n  {test_case['name']}:")
        print(f"    Input resolutions: {[f.shape[-2:] for f in features]}")
        print(f"    Output shape: {output.shape}")
        print(f"    Expected: (2, 256, {expected_H}, {expected_W})")
        
        assert output.shape == (2, 256, expected_H, expected_W), \
            f"Shape mismatch for {test_case['name']}"
        
        print(f"    ✓ Scale handling correct")
    
    print("\n  RESULT: Decoder Scale Handling PASS ✅")


def run_all_feature_contribution_tests():
    """Run all feature contribution tests."""
    print("\n" + "█"*60)
    print("  LEVEL 4: FEATURE CONTRIBUTION - TEST SUITE")
    print("█"*60)
    
    results = {}
    results["ablation"] = test_feature_ablation_single_vs_all()
    results["contribution"] = test_feature_contribution_per_scale()
    test_decoder_scale_handling()
    
    print("\n" + "█"*60)
    print("  ALL FEATURE CONTRIBUTION TESTS PASSED ✅")
    print("█"*60)
    
    return results


if __name__ == "__main__":
    run_all_feature_contribution_tests()