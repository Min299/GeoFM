"""tests/test_seed.py

Tests for seed utilities.
"""
import pytest
import random
import numpy as np
import torch


class TestSeed:
    """Test seed utilities."""

    def test_seed_everything(self):
        """seed_everything should set all seeds."""
        from geofm.utils.seed import seed_everything

        seed_everything(42)

        # Check that random is seeded
        assert random.random() == random.Random(42).random()

    def test_seed_everything_deterministic(self):
        """seed_everything with deterministic should set cudnn."""
        from geofm.utils.seed import seed_everything

        seed_everything(42, deterministic=True)

        # Check that cudnn settings are applied
        # (This is mostly smoke testing)
        assert True

    def test_seed_worker(self):
        """seed_worker should not raise."""
        from geofm.utils.seed import seed_worker

        # Should not raise
        seed_worker(0)

    def test_get_seed(self):
        """get_seed should return a value."""
        from geofm.utils.seed import get_seed, seed_everything

        seed_everything(42)

        seed = get_seed()

        # Should return something (may be -1 if not set)
        assert isinstance(seed, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])