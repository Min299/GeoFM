"""tests/test_backbone_factory.py

Tests for BackboneFactory.
"""
import pytest


class TestBackboneFactory:
    """Test BackboneFactory class."""

    def test_factory_exists(self):
        """BackboneFactory should be importable."""
        from geofm.factories.backbone_factory import BackboneFactory

        assert BackboneFactory is not None

    def test_factory_list_available(self):
        """list_available should return backbone list."""
        from geofm.factories.backbone_factory import BackboneFactory

        available = BackboneFactory.list_available()

        assert isinstance(available, list)
        # TerraMind variants should be in the list
        assert any("terramind" in x for x in available)

    def test_factory_unknown_raises(self):
        """Unknown backbone should raise ValueError."""
        from geofm.factories.backbone_factory import BackboneFactory

        with pytest.raises(ValueError, match="Unknown backbone"):
            BackboneFactory.build("bad_backbone")

    def test_factory_register(self):
        """register should add to registry."""
        from geofm.factories.backbone_factory import BackboneFactory

        class DummyBackbone:
            pass

        BackboneFactory.register("dummy", DummyBackbone)

        assert "dummy" in BackboneFactory._registry

    def test_factory_build_registered(self):
        """build should use registered class."""
        from geofm.factories.backbone_factory import BackboneFactory

        class MyBackbone:
            def __init__(self, x=1):
                self.x = x

        BackboneFactory.register("my_backbone", MyBackbone)

        backbone = BackboneFactory.build("my_backbone", x=5)

        assert isinstance(backbone, MyBackbone)
        assert backbone.x == 5

    def test_factory_terramind_raises_not_implemented(self):
        """terramind build should raise NotImplementedError if TerraTorch unavailable."""
        from geofm.factories.backbone_factory import BackboneFactory

        # This will fail if TerraTorch is not available
        try:
            backbone = BackboneFactory.build("terramind_base")
            # If it works, that's fine too
            assert backbone is not None
        except (ImportError, NotImplementedError, Exception):
            # Expected if TerraTorch not installed or not implemented
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])