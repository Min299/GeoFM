"""geofm.evaluation.regression_metrics

Regression metrics for damage scoring and other regression tasks.
"""
from __future__ import annotations

import torch


class RegressionMetrics:
    """Computes regression metrics (RMSE, MAE, etc.).

    Usage:
        metrics = RegressionMetrics()

        for batch in dataloader:
            pred, target = batch
            metrics.update(pred, target)

        results = metrics.compute()
        # {'rmse': 0.15, 'mae': 0.12, 'mse': 0.0225}
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()

    def reset(self):
        """Reset accumulated predictions and targets."""
        self.preds = []
        self.targets = []

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ):
        """Update metrics with a batch.

        Args:
            pred: Model predictions (B,) or (B, 1)
            target: Ground truth values (B,) or (B, 1)
        """
        # Flatten if needed
        if pred.dim() > 1:
            pred = pred.squeeze(-1)
        if target.dim() > 1:
            target = target.squeeze(-1)

        self.preds.append(pred.detach().cpu())
        self.targets.append(target.detach().cpu())

    def compute(self) -> dict:
        """Compute all metrics.

        Returns:
            Dictionary with RMSE, MAE, MSE, R2
        """
        pred = torch.cat(self.preds)
        target = torch.cat(self.targets)

        # MSE
        mse = ((pred - target) ** 2).mean()

        # RMSE
        rmse = mse.sqrt()

        # MAE
        mae = (pred - target).abs().mean()

        result = {
            "mse": mse.item(),
            "rmse": rmse.item(),
            "mae": mae.item(),
        }

        # R2 score
        ss_res = ((pred - target) ** 2).sum()
        ss_tot = ((target - target.mean()) ** 2).sum()

        if ss_tot > 0:
            r2 = 1 - (ss_res / ss_tot)
            result["r2"] = r2.item()
        else:
            result["r2"] = 0.0

        return result

    def __repr__(self):
        metrics = self.compute()
        return (
            f"RegressionMetrics("
            f"RMSE={metrics['rmse']:.4f}, "
            f"MAE={metrics['mae']:.4f})"
        )