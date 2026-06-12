"""tests/test_tiny_overfit.py

Test tiny dataset overfitting.
The gold standard: Can the model overfit 4 samples?

If the model cannot overfit 4 samples, there is an architecture bug.
"""
import pytest
import torch
import torch.nn as nn


class TestTinyOverfit:
    """Test tiny dataset overfitting."""

    def test_tiny_overfit_four_samples(self):
        """Model should overfit 4 samples."""
        from geofm.models.peft.task_feature_adapter import TaskFeatureAdapter

        # Create adapter
        adapter = TaskFeatureAdapter(feature_dims=[768, 768, 768, 768])

        # Create simple decoder (takes single tensor)
        decoder = nn.Conv2d(768, 2, 1)

        optimizer = torch.optim.Adam(
            list(adapter.parameters()) + list(decoder.parameters()),
            lr=0.01
        )
        criterion = nn.CrossEntropyLoss()

        # Create tiny dataset
        features = [torch.randn(4, 768, 64, 64) for _ in range(4)]
        targets = torch.randint(0, 2, (4, 64, 64))

        # Record initial loss
        with torch.no_grad():
            out = decoder(adapter(features)[0])
            initial_loss = criterion(out, targets).item()

        # Train for 50 iterations
        losses = []
        for i in range(50):
            optimizer.zero_grad()

            out = decoder(adapter(features)[0])
            loss = criterion(out, targets)

            losses.append(loss.item())

            loss.backward()
            optimizer.step()

        final_loss = losses[-1]

        # Loss should decrease significantly
        reduction = (initial_loss - final_loss) / initial_loss * 100

        assert final_loss < initial_loss, \
            f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"

        assert reduction > 30, \
            f"Loss reduction too small: {reduction:.1f}%"

        print(f"Tiny overfit: {initial_loss:.4f} -> {final_loss:.4f} ({reduction:.1f}% reduction)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
