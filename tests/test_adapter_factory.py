"""tests/test_adapter_factory.py

Tests for adapter factory.
"""
import pytest


class TestAdapterFactory:
    """Test AdapterFactory class."""

    def test_factory_exists(self):
        """AdapterFactory should be importable."""
        from geofm.factories import AdapterFactory

        assert AdapterFactory is not None

    def test_build_feature_adapter(self):
        """Should build feature adapter."""
        from geofm.factories import AdapterFactory

        adapter = AdapterFactory.build(
            "feature",
            dim=768,
            bottleneck_dim=64,
        )

        assert adapter is not None

    def test_build_task_feature_adapter(self):
        """Should build task feature adapter."""
        from geofm.factories import AdapterFactory

        adapter = AdapterFactory.build(
            "task_feature",
            feature_dims=[768, 768, 768, 768],
            bottleneck_dim=64,
        )

        assert adapter is not None

    def test_build_full_ft_returns_none(self):
        """Full ft should return None (no adapter)."""
        from geofm.factories import AdapterFactory

        adapter = AdapterFactory.build("full_ft")

        assert adapter is None

    def test_build_lora_raises(self):
        """LoRA should raise NotImplementedError."""
        from geofm.factories import AdapterFactory

        with pytest.raises(NotImplementedError):
            AdapterFactory.build("lora")

    def test_build_hybrid_raises(self):
        """Hybrid should raise NotImplementedError."""
        from geofm.factories import AdapterFactory

        with pytest.raises(NotImplementedError):
            AdapterFactory.build("hybrid")

    def test_unknown_adapter_raises(self):
        """Unknown adapter should raise ValueError."""
        from geofm.factories import AdapterFactory

        with pytest.raises(ValueError):
            AdapterFactory.build("unknown_adapter")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])