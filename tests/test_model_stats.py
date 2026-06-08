"""tests/test_model_stats.py

Tests for model statistics utilities.
"""
import pytest
import torch


class TestModelStats:
    """Test model statistics functions."""

    def test_count_parameters(self):
        """count_parameters should return correct count."""
        from geofm.utils.model_stats import count_parameters

        model = torch.nn.Linear(10, 5)

        # 10*5 weights + 5 bias = 55
        assert count_parameters(model) == 55

    def test_count_trainable_parameters(self):
        """count_trainable_parameters should count only trainable."""
        from geofm.utils.model_stats import count_trainable_parameters

        model = torch.nn.Linear(10, 5)

        assert count_trainable_parameters(model) == 55

    def test_count_trainable_parameters_with_freeze(self):
        """count_trainable_parameters should exclude frozen params."""
        from geofm.utils.model_stats import count_trainable_parameters

        model = torch.nn.Linear(10, 5)

        for p in model.parameters():
            p.requires_grad = False

        assert count_trainable_parameters(model) == 0

    def test_count_frozen_parameters(self):
        """count_frozen_parameters should count frozen params."""
        from geofm.utils.model_stats import count_frozen_parameters

        model = torch.nn.Linear(10, 5)

        for p in model.parameters():
            p.requires_grad = False

        assert count_frozen_parameters(model) == 55

    def test_get_model_summary(self):
        """get_model_summary should return all stats."""
        from geofm.utils.model_stats import get_model_summary

        model = torch.nn.Linear(10, 5)

        summary = get_model_summary(model)

        assert "total_parameters" in summary
        assert "trainable_parameters" in summary
        assert "frozen_parameters" in summary
        assert "trainable_percentage" in summary

        assert summary["total_parameters"] == 55
        assert summary["trainable_parameters"] == 55
        assert summary["frozen_parameters"] == 0
        assert summary["trainable_percentage"] == 100.0

    def test_print_model_summary(self):
        """print_model_summary should not raise."""
        from geofm.utils.model_stats import print_model_summary

        model = torch.nn.Linear(10, 5)

        # Should not raise
        print_model_summary(model, "TestModel")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])