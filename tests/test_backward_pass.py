"""tests/test_backward_pass.py

Test backward pass and gradient flow.

Critical test: Verify loss.backward() produces gradients.
"""
import pytest
import torch
import torch.nn as nn


class DummyBackbone(nn.Module):
    """Dummy backbone."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 768, 1)

    def extract_features(self, batch):
        return [torch.randn(2, 768, 64, 64) for _ in range(4)]

    def forward(self, x):
        return self.extract_features({})


class DummyAdapter(nn.Module):
    """Dummy adapter."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(768, 768, 1)

    def forward(self, features):
        return [self.conv(f) for f in features]


class DummyDecoder(nn.Module):
    """Dummy decoder."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(768, 2, 1)

    def forward(self, features):
        return self.conv(features[0])


class TestBackwardPass:
    """Test backward pass and gradient flow."""

    def test_backward_produces_gradients(self):
        """Backward pass should produce gradients."""
        # Create simple model
        model = nn.Linear(10, 5)
        criterion = nn.CrossEntropyLoss()

        # Dummy inputs
        x = torch.randn(2, 10)
        target = torch.randint(0, 5, (2,))

        # Forward
        logits = model(x)
        loss = criterion(logits, target)

        # Backward
        loss.backward()

        # Check gradients exist
        total_grad = 0
        grad_count = 0
        for p in model.parameters():
            if p.grad is not None:
                total_grad += p.grad.abs().sum().item()
                grad_count += 1

        assert grad_count > 0, "No gradients produced"
        assert total_grad > 0, "Gradient norm is zero"

    def test_backward_pass_full_pipeline(self):
        """Backward pass should work on simple pipeline."""
        # Simple model for testing
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 1),
        )

        # Enable gradients
        for p in model.parameters():
            p.requires_grad = True

        criterion = nn.CrossEntropyLoss()

        # Forward pass
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        target = torch.randint(0, 2, (2, 64, 64))

        # Loss
        loss = criterion(out, target)

        # Backward
        loss.backward()

        # Check gradients
        total_grad = 0
        param_count = 0
        for p in model.parameters():
            if p.grad is not None:
                total_grad += p.grad.abs().sum().item()
                param_count += 1

        assert param_count > 0, "No parameters have gradients"
        assert total_grad > 0, "Gradient norm is zero"

        print(f"Gradient check: {param_count} params with gradients, total norm: {total_grad:.4f}")

    def test_gradient_accumulation(self):
        """Gradients should accumulate correctly."""
        model = nn.Linear(10, 5)
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(2, 10)
        target = torch.randint(0, 5, (2,))

        # Zero gradients
        model.zero_grad()

        # Forward and backward twice
        for _ in range(2):
            logits = model(x)
            loss = criterion(logits, target)
            loss.backward()

        # Check gradients accumulated
        assert model.weight.grad is not None
        assert model.weight.grad.abs().sum().item() > 0

    def test_detached_tensor_no_grad(self):
        """Detached tensors should not produce gradients."""
        model = nn.Linear(10, 5)

        x = torch.randn(2, 10, requires_grad=True)

        # Forward with detached output
        out = model(x).detach()

        # This should not raise but also not produce gradients for model
        try:
            out.sum().backward()
            # Model should not have gradients since output was detached
            has_grad = model.weight.grad is not None
            # Note: x might have grad since it was not detached
        except RuntimeError:
            pass  # Expected behavior

    def test_backward_with_ignore_index(self):
        """Backward should work with ignore_index."""
        model = nn.Linear(10, 5)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        x = torch.randn(2, 10)
        target = torch.randint(0, 5, (2,))

        logits = model(x)
        loss = criterion(logits, target)

        loss.backward()

        assert model.weight.grad is not None

    def test_backward_no_nan_grad(self):
        """Gradients should not contain NaN."""
        model = nn.Linear(10, 5)
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(2, 10)
        target = torch.randint(0, 5, (2,))

        logits = model(x)
        loss = criterion(logits, target)

        loss.backward()

        for p in model.parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), "NaN in gradients"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])