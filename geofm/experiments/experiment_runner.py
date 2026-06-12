"""geofm.experiments.experiment_runner

Orchestrator for running experiments.
"""
from typing import Optional, Callable, Any
import torch
import logging

from .experiment import ExperimentConfig
from .experiment_builder import ExperimentBuilder
from .registry import ExperimentRegistry

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrator for running experiments.

    Usage:
        config = ExperimentConfig(
            name="exp01",
            task="flood",
            adaptation="lora",
        )
        trainer = SharedTrainer(...)
        runner = ExperimentRunner(config, trainer)
        model = runner.run()
    """

    def __init__(
        self,
        config: ExperimentConfig,
        trainer: Optional[Any] = None,
        evaluator: Optional[Any] = None,
        checkpoint_manager: Optional[Any] = None,
    ):
        """Initialize experiment runner.

        Args:
            config: Experiment configuration
            trainer: Training loop implementation
            evaluator: Evaluation implementation
            checkpoint_manager: Checkpoint save/load
        """
        self.config = config
        self.trainer = trainer
        self.evaluator = evaluator
        self.checkpoint_manager = checkpoint_manager

        self.model = None
        self.best_loss = float("inf")
        self.history = []

    def build_model(self) -> torch.nn.Module:
        """Build the model for this experiment.

        Returns:
            Built model
        """
        logger.info(f"Building model for {self.config.name}")
        logger.info(f"  Task: {self.config.task}")
        logger.info(f"  Adaptation: {self.config.adaptation}")
        logger.info(f"  Backbone: {self.config.backbone}")

        self.model = ExperimentBuilder.build(self.config)
        return self.model

    def setup(self):
        """Setup before training.

        - Set random seed
        - Build model
        - Setup optimizer if trainer not provided
        """
        # Set seed
        torch.manual_seed(self.config.epochs)  # Using epochs as seed fallback

        # Build model
        self.build_model()

        # Register experiment
        ExperimentRegistry.register(self.config.name, self.config)

        logger.info(f"Experiment '{self.config.name}' ready")

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Metrics dictionary
        """
        if self.trainer is None:
            raise RuntimeError("No trainer configured")

        metrics = self.trainer.train_epoch(self.model)

        self.history.append({
            "epoch": epoch,
            **metrics,
        })

        return metrics

    def validate(self) -> Optional[dict]:
        """Run validation.

        Returns:
            Validation metrics or None
        """
        if self.evaluator is None or self.trainer is None:
            return None

        if self.trainer.val_loader is None:
            return None

        return self.evaluator.evaluate(self.model, self.trainer.val_loader)

    def run(self) -> torch.nn.Module:
        """Run the full experiment.

        Returns:
            Trained model
        """
        self.setup()

        logger.info(f"Starting training for {self.config.epochs} epochs")

        for epoch in range(self.config.epochs):
            # Train
            metrics = self.train_epoch(epoch)

            # Log
            loss_str = f"Epoch {epoch}: loss={metrics.get('loss', 0):.4f}"
            if "val_loss" in metrics:
                loss_str += f", val_loss={metrics['val_loss']:.4f}"
            logger.info(loss_str)

            # Save checkpoint
            if self.checkpoint_manager and metrics.get("loss", 0) < self.best_loss:
                self.best_loss = metrics["loss"]
                self.save_checkpoint(f"best_{self.config.name}.pt")

        logger.info("Training complete")
        return self.model

    def save_checkpoint(self, path: str):
        """Save model checkpoint.

        Args:
            path: Checkpoint path
        """
        if self.checkpoint_manager is None:
            return

        self.checkpoint_manager.save(
            path,
            self.model,
            optimizer=getattr(self.trainer, "optimizer", None),
        )

    def load_checkpoint(self, path: str):
        """Load model checkpoint.

        Args:
            path: Checkpoint path
        """
        if self.checkpoint_manager is None:
            return

        self.checkpoint_manager.load(
            path,
            self.model,
            optimizer=getattr(self.trainer, "optimizer", None),
        )

    def get_history(self) -> list:
        """Get training history.

        Returns:
            List of epoch metrics
        """
        return self.history