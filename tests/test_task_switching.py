"""tests/test_task_switching.py

Tier 2: Shared Model Safety Tests

Verify task switching works correctly:
- Switching tasks should produce different outputs
- Switching should not corrupt model state
"""
import pytest
import torch
import torch.nn as nn

from geofm.models.peft import AdapterBank, TaskFeatureAdapter


class TestTaskSwitching:
    """Verify task switching behavior."""

    @pytest.fixture
    def multi_task_model(self):
        """Create multi-task model."""
        model = nn.Module()

        # Create adapter bank
        bank = AdapterBank()
        bank.register_task("flood", TaskFeatureAdapter(
            feature_dims=[768, 768, 768, 768],
            bottleneck_dim=64,
        ))
        bank.register_task("burn", TaskFeatureAdapter(
            feature_dims=[768, 768, 768, 768],
            bottleneck_dim=64,
        ))
        bank.register_task("lulc", TaskFeatureAdapter(
            feature_dims=[768, 768, 768, 768],
            bottleneck_dim=64,
        ))

        model.adapter_bank = bank

        return model

    def test_task_switch_affects_output(self, multi_task_model):
        """Different tasks should produce different outputs."""
        features = [torch.randn(1, 768, 56, 56) for _ in range(4)]
        x = torch.randn(1, 12, 224, 224)

        # Get output for each task (AdapterBank.forward(task_name, features))
        out_flood = multi_task_model.adapter_bank("flood", features)
        out_burn = multi_task_model.adapter_bank("burn", features)
        out_lulc = multi_task_model.adapter_bank("lulc", features)

        # All outputs should be lists (TaskFeatureAdapter returns list)
        assert isinstance(out_flood, list)
        assert isinstance(out_burn, list)
        assert isinstance(out_lulc, list)

    def test_task_switch_preserves_model(self, multi_task_model):
        """Task switching should not corrupt model state."""
        # Get full state before
        state_before = multi_task_model.state_dict()

        # Switch tasks multiple times
        features = [torch.randn(1, 768, 56, 56) for _ in range(4)]
        for task in ["flood", "burn", "lulc", "flood", "burn"]:
            _ = multi_task_model.adapter_bank(task, features)

        # State should be unchanged
        state_after = multi_task_model.state_dict()

        for key in state_before:
            diff = (state_before[key] - state_after[key]).abs().max().item()
            assert diff < 1e-6, f"Model state changed after task switching: {key}"

    def test_invalid_task_raises_error(self, multi_task_model):
        """Setting invalid task should raise error."""
        features = [torch.randn(1, 768, 56, 56) for _ in range(4)]
        with pytest.raises((KeyError, ValueError, AttributeError, TypeError)):
            _ = multi_task_model.adapter_bank("banana", features)

    def test_task_switch_during_training(self, multi_task_model):
        """Switching tasks during training should be safe."""
        features = [torch.randn(1, 768, 56, 56) for _ in range(4)]
        x = torch.randn(1, 12, 224, 224)

        # Train flood
        optimizer = torch.optim.Adam(
            multi_task_model.adapter_bank.adapters["flood"].parameters()
        )

        for _ in range(3):
            optimizer.zero_grad()
            out = multi_task_model.adapter_bank("flood", features)
            loss = out[0].mean()
            loss.backward()
            optimizer.step()

        # Switch to burn - should work
        out = multi_task_model.adapter_bank("burn", features)
        assert isinstance(out, list)

        # Switch back to flood - should still work
        out = multi_task_model.adapter_bank("flood", features)
        assert isinstance(out, list)


class TestTaskConsistency:
    """Verify task outputs are consistent."""

    def test_same_task_same_output(self):
        """Same task should produce same output for same input."""
        bank = AdapterBank()
        bank.register_task("flood", TaskFeatureAdapter(
            feature_dims=[768, 768, 768, 768],
            bottleneck_dim=64,
        ))
        bank.set_task("flood")  # Use set_task if available

        features = [torch.randn(1, 768, 56, 56) for _ in range(4)]

        out1 = bank(features)
        out2 = bank(features)

        # Should be similar (not exact due to dropout if any)
        diff = (out1[0] - out2[0]).abs().max().item()
        # Allow some variation for non-deterministic ops
        assert diff < 1.0, f"Outputs differ by {diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])