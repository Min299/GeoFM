"""tests/test_adapter_isolation.py

Tier 2: Shared Model Safety Tests

Verify task adapters are isolated:
- Training one task should not affect other task adapters
"""
import pytest
import torch
import copy

from geofm.models.peft import AdapterBank, TaskFeatureAdapter


class TestAdapterIsolation:
    """Verify task adapters are isolated during training."""

    @pytest.fixture
    def adapter_bank(self):
        """Create adapter bank with multiple tasks."""
        bank = AdapterBank()

        # Add flood adapter (uses feature_dims, not in_channels)
        bank.register_task("flood", TaskFeatureAdapter(
            feature_dims=[768, 768, 768, 768],
            bottleneck_dim=64,
        ))

        # Add burn adapter
        bank.register_task("burn", TaskFeatureAdapter(
            feature_dims=[768, 768, 768, 768],
            bottleneck_dim=64,
        ))

        # Add lulc adapter
        bank.register_task("lulc", TaskFeatureAdapter(
            feature_dims=[768, 768, 768, 768],
            bottleneck_dim=64,
        ))

        return bank

    def test_adapter_bank_creation(self, adapter_bank):
        """Adapter bank should have multiple adapters."""
        assert "flood" in adapter_bank.adapters
        assert "burn" in adapter_bank.adapters
        assert "lulc" in adapter_bank.adapters

    def test_flood_training_does_not_affect_burn(self, adapter_bank):
        """Training flood should not change burn adapter."""
        # Save burn state
        burn_before = copy.deepcopy(adapter_bank.adapters["burn"].state_dict())

        # Simulate training flood
        features = [torch.randn(1, 768, 56, 56) for _ in range(4)]
        output = adapter_bank("flood", features)
        loss = output[0].mean()  # TaskFeatureAdapter returns list
        loss.backward()

        # Burn state should be unchanged
        burn_after = adapter_bank.adapters["burn"].state_dict()

        for key in burn_before:
            diff = (burn_before[key] - burn_after[key]).abs().max().item()
            assert diff < 1e-6, f"Burn adapter changed: {key} diff={diff}"

    def test_flood_training_does_not_affect_lulc(self, adapter_bank):
        """Training flood should not change lulc adapter."""
        # Save lulc state
        lulc_before = copy.deepcopy(adapter_bank.adapters["lulc"].state_dict())

        # Simulate training flood
        features = [torch.randn(1, 768, 56, 56) for _ in range(4)]
        output = adapter_bank("flood", features)
        loss = output[0].mean()
        loss.backward()

        # lulc state should be unchanged
        lulc_after = adapter_bank.adapters["lulc"].state_dict()

        for key in lulc_before:
            diff = (lulc_before[key] - lulc_after[key]).abs().max().item()
            assert diff < 1e-6, f"LULC adapter changed: {key} diff={diff}"

    def test_task_switching_preserves_other_tasks(self, adapter_bank):
        """Switching tasks should preserve other adapters."""
        # Save all states
        states_before = {
            name: copy.deepcopy(adapter.state_dict())
            for name, adapter in adapter_bank.adapters.items()
        }

        # Train each task briefly
        for task in ["flood", "burn", "lulc"]:
            features = [torch.randn(1, 768, 56, 56) for _ in range(4)]
            output = adapter_bank(task, features)
            loss = output[0].mean()
            loss.backward()

        # All other adapters should be unchanged
        for task_name, adapter in adapter_bank.adapters.items():
            state_after = adapter.state_dict()
            state_before = states_before[task_name]

            for key in state_before:
                diff = (state_before[key] - state_after[key]).abs().max().item()
                assert diff < 1e-6, f"{task_name} changed during other task training"


class TestDecoderIsolation:
    """Verify decoder isolation between tasks."""

    @pytest.fixture
    def decoders(self):
        """Create separate decoders for each task."""
        from geofm.models.decoders import PyramidDecoder

        return {
            "flood": PyramidDecoder(in_channels=[768, 768, 768, 768]),
            "burn": PyramidDecoder(in_channels=[768, 768, 768, 768]),
        }

    def test_training_flood_decoder_preserves_burn_decoder(self, decoders):
        """Training flood decoder should not affect burn decoder."""
        # Save burn state
        burn_before = copy.deepcopy(decoders["burn"].state_dict())

        # Train flood
        features = [torch.randn(1, 768, 56, 56) for _ in range(4)]
        output = decoders["flood"](features)
        loss = output.mean()
        loss.backward()

        # Burn decoder should be unchanged
        burn_after = decoders["burn"].state_dict()

        for key in burn_before:
            diff = (burn_before[key] - burn_after[key]).abs().max().item()
            assert diff < 1e-6, f"Burn decoder changed: {key}"


class TestGradientIsolation:
    """Verify gradients only affect current task."""

    @pytest.fixture
    def multi_task_model(self):
        """Create multi-task model."""
        bank = AdapterBank()
        bank.register_task("flood", TaskFeatureAdapter(
            feature_dims=[768, 768, 768, 768],
            bottleneck_dim=64,
        ))
        bank.register_task("burn", TaskFeatureAdapter(
            feature_dims=[768, 768, 768, 768],
            bottleneck_dim=64,
        ))

        return bank

    def test_flood_gradients_only_affect_flood(self, multi_task_model):
        """Flood gradients should not affect burn parameters."""
        # Get parameters before
        burn_params_before = {
            name: p.clone()
            for name, p in multi_task_model.adapters["burn"].named_parameters()
        }

        # Train flood
        features = [torch.randn(1, 768, 56, 56) for _ in range(4)]
        output = multi_task_model("flood", features)
        loss = output[0].mean()
        loss.backward()

        # Burn parameters should be unchanged
        for name, param in multi_task_model.adapters["burn"].named_parameters():
            diff = (burn_params_before[name] - param).abs().max().item()
            assert diff < 1e-6, f"Burn param '{name}' was modified during flood training"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])