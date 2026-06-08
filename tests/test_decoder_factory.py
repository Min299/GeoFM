"""tests/test_decoder_factory.py

Tests for decoder factory.
"""
import pytest


class TestDecoderFactory:
    """Test DecoderFactory class."""

    def test_factory_exists(self):
        """DecoderFactory should be importable."""
        from geofm.factories import DecoderFactory

        assert DecoderFactory is not None

    def test_build_flood_decoder(self):
        """Should build flood decoder."""
        from geofm.factories import DecoderFactory

        decoder = DecoderFactory.build("flood")

        assert decoder is not None

    def test_build_burn_decoder(self):
        """Should build burn decoder."""
        from geofm.factories import DecoderFactory

        decoder = DecoderFactory.build("burn")

        assert decoder is not None

    def test_build_lulc_decoder(self):
        """Should build LULC decoder."""
        from geofm.factories import DecoderFactory

        decoder = DecoderFactory.build("lulc")

        assert decoder is not None

    def test_unknown_task_raises(self):
        """Unknown task should raise ValueError."""
        from geofm.factories import DecoderFactory

        with pytest.raises(ValueError):
            DecoderFactory.build("unknown_task")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])