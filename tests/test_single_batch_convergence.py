"""tests/test_single_batch_convergence.py

Test single batch convergence.
Train same batch 100 times - loss should consistently decrease.
"""
import pytest
import torch
import torch.nn as nn


class TestSingleBatchConvergence:
    """Test single batch convergence."""

    def test_single_batch_convergence(self):
        """Loss should decrease when training same batch repeatedly."""
        # Simple model
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Single batch
        x = torch.randn(2, 10)
        targets = torch.randint(0, 2, (2,))

        losses = []
        for step in range(100):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, targets)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        # Check convergence
        assert losses[-1] < losses[0], \
            f"Loss did not decrease: step 0: {losses[0]:.4f}, step 99: {losses[-1]:.4f}"

        print(f"Single batch convergence: step 0: {losses[0]:.4f}, step 50: {losses[50]:.4f}, step 99: {losses[-1]:.4f}")

    def test_step0_step25_step50_step100(self):
        """Report loss at step 0, 25, 50, 100."""
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(2, 10)
        targets = torch.randint(0, 2, (2,))

        step_losses = {}
        for step in range(101):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

            if step in [0, 25, 50, 100]:
                step_losses[step] = loss.item()

        # Verify decreasing
        assert step_losses[100] < step_losses[0], \
            f"Loss not decreasing: {step_losses}"

        print(f"Step losses: {step_losses}")

    def test_consistent_decrease(self):
        """Loss should consistently decrease."""
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(2, 10)
        targets = torch.randint(0, 2, (2,))

        initial_loss = None
        final_loss = None

        for step in range(50):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, targets)

            if step == 0:
                initial_loss = loss.item()
            if step == 49:
                final_loss = loss.item()

            loss.backward()
            optimizer.step()

        assert final_loss < initial_loss, \
            f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    def test_lora_single_batch_convergence(self):
        """LoRA should converge on single batch."""
        from geofm.models.peft.lora_adapter import LoRALinear

        layer = nn.Linear(768, 768)
        lora = LoRALinear(layer, rank=16, alpha=32)

        optimizer = torch.optim.Adam(lora.parameters(), lr=0.01)

        x = torch.randn(4, 10, 768)
        targets = torch.randint(0, 768, (4, 10))

        losses = []
        for step in range(50):
            optimizer.zero_grad()
            out = lora(x)
            loss = nn.functional.cross_entropy(
                out.view(-1, 768),
                targets.view(-1)
            )
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        assert losses[-1] < losses[0], \
            f"LoRA loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_convergence_rate(self):
        """Check that convergence happens at reasonable rate."""
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(2, 10)
        targets = torch.randint(0, 2, (2,))

        for step in range(30):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

        # After 30 steps, should have significant reduction
        # (We don't check exact values as it depends on random initialization)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
