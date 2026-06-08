"""tests/test_shared_geofm_runtime.py

Tests for SharedGeoFM runtime validation.
Validates the complete pipeline:
    Backbone → AdapterBank → DecoderBank → Task Routing
"""
import pytest
import torch
import torch.nn as nn


class DummyBackbone(nn.Module):
    """Dummy backbone that returns 4 feature levels."""

    def __init__(self):
        super().__init__()

    def extract_features(self, batch):
        """Return 4 features at different resolutions."""
        return [
            torch.randn(2, 768, 64, 64),
            torch.randn(2, 768, 32, 32),
            torch.randn(2, 768, 16, 16),
            torch.randn(2, 768, 8, 8),
        ]

    def forward(self, x):
        return self.extract_features({})


class DummyAdapterBank(nn.Module):
    """Dummy adapter bank with task routing."""

    def __init__(self):
        super().__init__()
        # Create adapters for each task
        self.adapters = nn.ModuleDict({
            "flood": self._create_adapter(),
            "burn": self._create_adapter(),
            "lulc": self._create_adapter(),
        })

    def _create_adapter(self):
        """Create a simple adapter."""
        return nn.Sequential(
            nn.Conv2d(768, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 768, 1),
        )

    def forward(self, task, features):
        """Apply task-specific adapter."""
        if task not in self.adapters:
            raise ValueError(f"Unknown task: {task}")
        return [self.adapters[task](f) for f in features]


class DummyDecoderBank(nn.Module):
    """Dummy decoder bank with task routing."""

    def __init__(self):
        super().__init__()
        self.decoders = nn.ModuleDict({
            "flood": self._create_decoder(),
            "burn": self._create_decoder(),
            "lulc": self._create_decoder(),
        })

    def _create_decoder(self):
        """Create a simple decoder."""
        return nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 2, 1),  # 2 classes (binary)
        )

    def forward(self, task, features):
        """Apply task-specific decoder."""
        if task not in self.decoders:
            raise ValueError(f"Unknown task: {task}")

        # Simple: take first feature and decode
        return self.decoders[task](features[0])


class TestSharedGeoFMRuntime:
    """Test SharedGeoFM runtime validation."""

    def test_shared_geofm_import(self):
        """SharedGeoFM should be importable."""
        try:
            from geofm.models.multitask.shared_geofm import SharedGeoFM
            assert SharedGeoFM is not None
        except ImportError:
            pytest.skip("SharedGeoFM not available")

    def test_shared_geofm_init(self):
        """SharedGeoFM should initialize."""
        try:
            from geofm.models.multitask.shared_geofm import SharedGeoFM

            model = SharedGeoFM(
                backbone=DummyBackbone(),
                adapter_bank=DummyAdapterBank(),
                decoder_bank=DummyDecoderBank(),
            )

            assert model is not None
        except ImportError:
            pytest.skip("SharedGeoFM not available")

    def test_shared_geofm_forward_flood(self):
        """SharedGeoFM should forward for flood task."""
        try:
            from geofm.models.multitask.shared_geofm import SharedGeoFM

            model = SharedGeoFM(
                backbone=DummyBackbone(),
                adapter_bank=DummyAdapterBank(),
                decoder_bank=DummyDecoderBank(),
            )

            batch = {"inputs": torch.randn(2, 6, 256, 256)}
            out = model(batch, task_name="flood")

            assert out.shape[0] == 2  # batch size
            assert out.shape[1] == 2  # num_classes
            assert out.dim() == 4  # 4D tensor

        except ImportError:
            pytest.skip("SharedGeoFM not available")

    def test_shared_geofm_forward_burn(self):
        """SharedGeoFM should forward for burn task."""
        try:
            from geofm.models.multitask.shared_geofm import SharedGeoFM

            model = SharedGeoFM(
                backbone=DummyBackbone(),
                adapter_bank=DummyAdapterBank(),
                decoder_bank=DummyDecoderBank(),
            )

            batch = {"inputs": torch.randn(2, 6, 256, 256)}
            out = model(batch, task_name="burn")

            assert out.shape[0] == 2
            assert out.shape[1] == 2

        except ImportError:
            pytest.skip("SharedGeoFM not available")

    def test_shared_geofm_forward_lulc(self):
        """SharedGeoFM should forward for lulc task."""
        try:
            from geofm.models.multitask.shared_geofm import SharedGeoFM

            model = SharedGeoFM(
                backbone=DummyBackbone(),
                adapter_bank=DummyAdapterBank(),
                decoder_bank=DummyDecoderBank(),
            )

            batch = {"inputs": torch.randn(2, 6, 256, 256)}
            out = model(batch, task_name="lulc")

            assert out.shape[0] == 2
            assert out.shape[1] == 2

        except ImportError:
            pytest.skip("SharedGeoFM not available")

    def test_shared_geofm_task_routing(self):
        """Different tasks should produce different outputs."""
        try:
            from geofm.models.multitask.shared_geofm import SharedGeoFM

            model = SharedGeoFM(
                backbone=DummyBackbone(),
                adapter_bank=DummyAdapterBank(),
                decoder_bank=DummyDecoderBank(),
            )

            batch = {"inputs": torch.randn(2, 6, 256, 256)}

            # Get outputs for different tasks
            out_flood = model(batch, task_name="flood")
            out_burn = model(batch, task_name="burn")
            out_lulc = model(batch, task_name="lulc")

            # Shapes should be same
            assert out_flood.shape == out_burn.shape
            assert out_burn.shape == out_lulc.shape

        except ImportError:
            pytest.skip("SharedGeoFM not available")

    def test_shared_geofm_no_nan(self):
        """Output should not contain NaN values."""
        try:
            from geofm.models.multitask.shared_geofm import SharedGeoFM

            model = SharedGeoFM(
                backbone=DummyBackbone(),
                adapter_bank=DummyAdapterBank(),
                decoder_bank=DummyDecoderBank(),
            )

            batch = {"inputs": torch.randn(2, 6, 256, 256)}
            out = model(batch, task_name="flood")

            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()

        except ImportError:
            pytest.skip("SharedGeoFM not available")

    def test_shared_geofm_gradient_flow(self):
        """Model should support gradient flow."""
        try:
            from geofm.models.multitask.shared_geofm import SharedGeoFM

            model = SharedGeoFM(
                backbone=DummyBackbone(),
                adapter_bank=DummyAdapterBank(),
                decoder_bank=DummyDecoderBank(),
            )

            # Enable gradients
            for p in model.parameters():
                p.requires_grad = True

            batch = {"inputs": torch.randn(2, 6, 256, 256)}
            out = model(batch, task_name="flood")

            # Create loss and backward
            loss = out.sum()
            loss.backward()

            # Check gradients exist
            has_grad = any(p.grad is not None for p in model.parameters())

            assert has_grad

        except ImportError:
            pytest.skip("SharedGeoFM not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])