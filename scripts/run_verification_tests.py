#!/usr/bin/env python3
"""Run all verification tests.

This script runs the complete verification test suite and reports results.
"""
import subprocess
import sys
import os

# Test files to run
TEST_FILES = [
    # Tier 1: Architecture Integrity
    "tests/test_lora_injection_verification.py",
    "tests/test_parameter_freezing.py",
    "tests/test_hybrid_path.py",
    
    # Tier 2: Shared Model Safety
    "tests/test_adapter_isolation.py",
    "tests/test_task_switching.py",
    
    # Tier 3: Decoder Verification
    "tests/test_feature_contribution.py",
    
    # Tier 4: Experiment System
    "tests/test_experiment_builder.py",
    "tests/test_checkpoint_roundtrip.py",
    
    # Tier 5: End-to-End Smoke Tests
    "tests/test_end_to_end_smoke.py",
    
    # Edge Cases
    "tests/test_edge_cases.py",
]

def run_tests():
    """Run all test files."""
    os.chdir("/workspace/project/GeoFM")
    
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    
    print("=" * 70)
    print(" GEOFM VERIFICATION TEST SUITE")
    print("=" * 70)
    print()
    
    for test_file in TEST_FILES:
        if not os.path.exists(test_file):
            print(f"⚠ SKIP: {test_file} (not found)")
            continue
            
        print(f"\n{'=' * 70}")
        print(f" Running: {test_file}")
        print("=" * 70)
        
        result = subprocess.run(
            ["python", "-m", "pytest", test_file, "-v", "--tb=short"],
            env={**os.environ, "PYTHONPATH": "/workspace/project/GeoFM"},
            capture_output=False,
        )
        
        if result.returncode == 0:
            print(f"✅ PASSED: {test_file}")
            total_passed += 1
        elif result.returncode == 5:
            # No tests collected
            print(f"⚠ NO TESTS: {test_file}")
            total_skipped += 1
        else:
            print(f"❌ FAILED: {test_file}")
            total_failed += 1
    
    print()
    print("=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f"  Passed:  {total_passed}")
    print(f"  Failed: {total_failed}")
    print(f"  Skipped: {total_skipped}")
    print("=" * 70)
    
    return total_failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)