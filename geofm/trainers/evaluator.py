"""geofm.trainers.evaluator

Evaluation metrics for GeoFM experiments.
"""
import torch
import torch.nn.functional as F
from typing import Dict, Optional


class Evaluator:
    """Evaluator for model predictions.

    Provides metrics for segmentation and classification tasks.

    Usage:
        evaluator = Evaluator()
        metrics = evaluator.evaluate(model, dataloader)
    """

    @staticmethod
    def accuracy(
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        """Calculate accuracy.

        Args:
            pred: Predictions (B, C, ...) or (B, ...)
            target: Ground truth (B, ...) or (B, ...)

        Returns:
            Accuracy score
        """
        if pred.dim() > 1 and pred.shape[1] > 1:
            pred = pred.argmax(dim=1)

        correct = (pred == target).float().mean().item()
        return correct

    @staticmethod
    def iou(
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: Optional[int] = None,
    ) -> float:
        """Calculate Intersection over Union (IoU).

        Args:
            pred: Predicted class indices (B, H, W)
            target: Ground truth class indices (B, H, W)
            num_classes: Number of classes

        Returns:
            Mean IoU score
        """
        if num_classes is None:
            num_classes = max(pred.max().item(), target.max().item()) + 1

        ious = []

        for cls in range(num_classes):
            pred_cls = (pred == cls)
            target_cls = (target == cls)

            intersection = (pred_cls & target_cls).sum().item()
            union = (pred_cls | target_cls).sum().item()

            if union > 0:
                ious.append(intersection / union)
            else:
                ious.append(float("nan"))

        # Return mean IoU (ignoring NaN)
        valid_ious = [x for x in ious if not torch.isnan(torch.tensor(x))]
        if valid_ious:
            return sum(valid_ious) / len(valid_ious)
        return 0.0

    @staticmethod
    def dice(
        pred: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5,
    ) -> float:
        """Calculate Dice coefficient.

        Args:
            pred: Predictions (B, C, H, W) or (B, H, W)
            target: Ground truth (B, H, W)
            threshold: Threshold for binary predictions

        Returns:
            Dice score
        """
        # Apply sigmoid/softmax if needed
        if pred.dim() == 4 and pred.shape[1] > 1:
            pred = F.softmax(pred, dim=1).argmax(dim=1)
        elif pred.dim() == 4 and pred.shape[1] == 1:
            pred = (pred > threshold).squeeze(1).float()

        intersection = (pred * target).sum()
        dice = (2.0 * intersection + 1e-7) / (pred.sum() + target.sum() + 1e-7)

        return dice.item()

    @staticmethod
    def f1_score(
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        """Calculate F1 score.

        Args:
            pred: Predicted class indices (B, H, W)
            target: Ground truth class indices (B, H, W)

        Returns:
            F1 score
        """
        tp = ((pred == 1) & (target == 1)).sum().item()
        fp = ((pred == 1) & (target == 0)).sum().item()
        fn = ((pred == 0) & (target == 1)).sum().item()

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)

        f1 = 2 * precision * recall / (precision + recall + 1e-7)

        return f1

    @staticmethod
    def rmse(
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        """Calculate Root Mean Square Error.

        Args:
            pred: Predictions
            target: Ground truth

        Returns:
            RMSE score
        """
        mse = F.mse_loss(pred, target)
        rmse = torch.sqrt(mse).item()
        return rmse

    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader,
        metrics: Optional[list] = None,
    ) -> Dict[str, float]:
        """Evaluate model on a dataloader.

        Args:
            model: Model to evaluate
            dataloader: Data loader
            metrics: List of metrics to compute

        Returns:
            Dictionary of metric scores
        """
        if metrics is None:
            metrics = ["accuracy", "iou"]

        model.eval()
        results = {}

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    inputs = batch.get("image", batch.get("x"))
                    targets = batch.get("target", batch.get("labels"))
                else:
                    inputs, targets = batch

                outputs = model(inputs)

                if "accuracy" in metrics:
                    all_preds.append(outputs.argmax(dim=1) if outputs.dim() > 1 else outputs)
                    all_targets.append(targets)

        if all_preds:
            preds = torch.cat(all_preds, dim=0)
            targets = torch.cat(all_targets, dim=0)

            if "accuracy" in metrics:
                results["accuracy"] = self.accuracy(preds, targets)
            if "iou" in metrics:
                results["iou"] = self.iou(preds, targets)
            if "f1" in metrics:
                results["f1"] = self.f1_score(preds, targets)

        return results


class SegmentationEvaluator(Evaluator):
    """Evaluator specialized for segmentation tasks."""

    @staticmethod
    def pixel_accuracy(
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        """Calculate pixel-wise accuracy.

        Args:
            pred: Predicted segmentation (B, H, W)
            target: Ground truth (B, H, W)

        Returns:
            Pixel accuracy
        """
        correct = (pred == target).sum().item()
        total = target.numel()
        return correct / total

    @staticmethod
    def miou(
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
    ) -> float:
        """Calculate mean IoU across all classes.

        Args:
            pred: Predicted class indices (B, H, W)
            target: Ground truth class indices (B, H, W)
            num_classes: Number of classes

        Returns:
            Mean IoU
        """
        ious = []

        for cls in range(num_classes):
            pred_cls = (pred == cls)
            target_cls = (target == cls)

            intersection = (pred_cls & target_cls).sum().item()
            union = (pred_cls | target_cls).sum().item()

            if union > 0:
                ious.append(intersection / union)

        return sum(ious) / num_classes if ious else 0.0


class RegressionEvaluator(Evaluator):
    """Evaluator specialized for regression tasks."""

    @staticmethod
    def mae(
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        """Calculate Mean Absolute Error.

        Args:
            pred: Predictions
            target: Ground truth

        Returns:
            MAE score
        """
        return F.l1_loss(pred, target).item()

    @staticmethod
    def r2_score(
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        """Calculate R² score.

        Args:
            pred: Predictions
            target: Ground truth

        Returns:
            R² score
        """
        target_mean = target.mean()
        ss_tot = ((target - target_mean) ** 2).sum()
        ss_res = ((target - pred) ** 2).sum()

        if ss_tot > 0:
            return 1 - (ss_res / ss_tot)
        return 0.0
