"""tests/test_optimizer_updates.py

Test that optimizer actually updates parameters.
Critical: Verify weights change after optimizer.step().
"""
import pytest
import torch
import torch.nn as nn


class TestOptimizerUpdates:
    """Test optimizer parameter updates."""

    def test_weights_change_after_step(self):
        """Weights should change after optimizer.step()."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        criterion = nn.CrossEntropyLoss()

        x = torch.randn(2, 10)
        target = torch.randint(0, 5, (2,))

        # Store weights before
        weight_before = model.weight.clone()

        # Forward, backward, step
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        # Store weights after
        weight_after = model.weight

        # Weights should be different
        assert not torch.allclose(weight_before, weight_after, atol=1e-6), \
            "Weights did not change after optimizer.step()"

    def test_adapter_weights_change(self):
        """Adapter weights should change during training."""
        from geofm.models.peft.feature_adapter import FeatureAdapter

        adapter = FeatureAdapter(dim=768, bottleneck_dim=64)
        optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)

        x = torch.randn(2, 768, 64, 64)

        # Store first weight
        for name, param in adapter.named_parameters():
            if 'down' in name:
                down_before = param.clone()
                break

        # Forward, backward, step
        optimizer.zero_grad()
        out = adapter(x)
        loss = out[0].sum()
        loss.backward()
        optimizer.step()

        # Check weight changed
        for name, param in adapter.named_parameters():
            if 'down' in name:
                assert not torch.allclose(down_before, param, atol=1e-6), \
                    f"Adapter down weights did not change"
                break

    def test_lora_weights_change(self):
        """LoRA weights should change during training."""
        from geofm.models.peft.lora_adapter import LoRALinear

        layer = nn.Linear(768, 768)
        lora = LoRALinear(layer, rank=16, alpha=32)

        optimizer = torch.optim.SGD(lora.parameters(), lr=0.1)

        x = torch.randn(2, 10, 768)

        # Store lora_B before
        lora_B_before = lora.lora_B.clone()

        # Forward, backward, step
        optimizer.zero_grad()
        out = lora(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        # LoRA B should change
        assert not torch.allclose(lora_B_before, lora.lora_B, atol=1e-6), \
            "LoRA B weights did not change"

    def test_multiple_steps_accumulate_change(self):
        """Multiple steps should accumulate parameter change."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        criterion = nn.CrossEntropyLoss()

        x = torch.randn(2, 10)
        target = torch.randint(0, 5, (2,))

        # Store initial weights
        weight_initial = model.weight.clone()

        # Take 10 steps
        for _ in range(10):
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

        # Weight should be different from initial
        assert not torch.allclose(weight_initial, model.weight, atol=1e-4), \
            "Weights did not change after multiple steps"

    def test_frozen_params_do_not_change(self):
        """Frozen params should not change during training."""
        backbone = nn.Linear(10, 10)
        adapter = nn.Linear(10, 5)

        # Freeze backbone
        for p in backbone.parameters():
            p.requires_grad = False

        model = nn.Sequential(backbone, adapter)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Store backbone weights
        backbone_weight_before = backbone.weight.clone()

        # Forward, backward, step
        x = torch.randn(2, 10)
        target = torch.randint(0, 5, (2,))

        optimizer.zero_grad()
        out = model(x)
        loss = nn.functional.cross_entropy(out, target)
        loss.backward()
        optimizer.step()

        # Frozen weights should not change
        assert torch.allclose(backbone_weight_before, backbone.weight, atol=1e-6), \
            "Frozen weights changed unexpectedly"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
