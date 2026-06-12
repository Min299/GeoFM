"""geofm.training.fit

High-level training loop combining epoch execution with history tracking.
"""
from __future__ import annotations


def fit(
    runner,
    train_loader,
    val_loader,
    epochs,
    logger=None,
    task: str = "flood",
):
    """Train a model for multiple epochs.

    Args:
        runner: EpochRunner instance
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs to train
        logger: Optional logger with log() method
        task: Task name for model forward pass

    Returns:
        Dictionary with 'train_loss' and 'val_loss' history lists
    """
    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(epochs):
        train_loss = runner.train_epoch(
            train_loader,
            task=task,
        )

        val_loss = runner.validate_epoch(
            val_loader,
            task=task,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if logger:
            logger.log(
                f"epoch={epoch} "
                f"train={train_loss:.4f} "
                f"val={val_loss:.4f}"
            )

    return history