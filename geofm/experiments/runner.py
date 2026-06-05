"""geofm.experiments.runner

Experiment runner for training and evaluation.
"""
from typing import Optional, Dict, Any
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from geofm.experiments.experiment import ExperimentConfig, ExperimentResult
from geofm.experiments.registry import get_registry


class ExperimentRunner:
    """Runs experiments and tracks results.

    Usage:
        runner = ExperimentRunner()
        result = runner.run(config, model, train_loader, val_loader)
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.metrics_dir = self.output_dir / "metrics"
        self.visualization_dir = self.output_dir / "visualizations"

        # Create directories
        for d in [
            self.checkpoint_dir,
            self.log_dir,
            self.metrics_dir,
            self.visualization_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        self.registry = get_registry()

    def run(
        self,
        config: ExperimentConfig,
        model: torch.nn.Module,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        trainer: Optional[Any] = None,
    ) -> ExperimentResult:
        """Run an experiment.

        Args:
            config: Experiment configuration
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            trainer: Custom trainer (if None, uses default)

        Returns:
            ExperimentResult with metrics
        """
        print(f"Starting experiment: {config.name}")
        print(f"  Task: {config.task}")
        print(f"  Mode: {config.training_mode}")
        print(f"  Backbone: {config.backbone}")

        # Register experiment
        self.registry.register(config)

        # Create experiment output directory
        exp_dir = self.output_dir / config.name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config.save(exp_dir / "config.json")

        # Initialize trainer
        if trainer is None:
            from geofm.trainers import FineTuneTrainer
            trainer = FineTuneTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device="cuda" if torch.cuda.is_available() else "cpu",
                max_epochs=config.epochs,
            )

        # Train
        start_time = time.time()
        history = trainer.train()
        duration = time.time() - start_time

        # Get best metrics
        best_epoch = history["val_loss"].index(min(history["val_loss"]))
        best_val_loss = min(history["val_loss"])

        # Final validation
        final_metrics = trainer.validate()

        # Create result
        result = ExperimentResult(
            config=config,
            metrics={
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "train_loss_final": history["train_loss"][-1] if history["train_loss"] else 0,
                "val_loss_final": history["val_loss"][-1] if history["val_loss"] else 0,
                "history": {
                    "train_loss": history["train_loss"],
                    "val_loss": history["val_loss"],
                },
            },
            best_epoch=best_epoch,
            total_epochs=len(history["train_loss"]),
            duration_seconds=duration,
            checkpoint_path=str(exp_dir / "checkpoint.pt"),
        )

        # Save checkpoint
        trainer.save_checkpoint(str(exp_dir / "checkpoint.pt"))

        # Save results
        result.save(exp_dir / "results.json")

        # Register result
        self.registry.add_result(result)

        print(f"Experiment complete: {config.name}")
        print(f"  Best epoch: {best_epoch}")
        print(f"  Best val loss: {best_val_loss:.4f}")
        print(f"  Duration: {duration:.1f}s")

        return result

    def run_comparison(
        self,
        task: str,
        base_model: torch.nn.Module,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        **kwargs
    ) -> Dict[str, ExperimentResult]:
        """Run comparison between Full FT and LoRA.

        Args:
            task: Task name
            base_model: Base model to use
            train_loader: Training data loader
            val_loader: Validation data loader
            **kwargs: Additional config parameters

        Returns:
            Dict of mode -> ExperimentResult
        """
        from geofm.models.lora import inject_lora

        results = {}

        # Full FT
        print("\n" + "=" * 50)
        print("Running Full FT experiment...")
        print("=" * 50)

        full_ft_config = ExperimentConfig(
            name=f"{task}_full_ft",
            task=task,
            training_mode="full_ft",
            **kwargs
        )
        results["full_ft"] = self.run(
            full_ft_config,
            base_model,
            train_loader,
            val_loader,
        )

        # LoRA
        print("\n" + "=" * 50)
        print("Running LoRA experiment...")
        print("=" * 50)

        lora_model = inject_lora(
            base_model,
            rank=kwargs.get("lora_rank", 16),
            alpha=kwargs.get("lora_alpha", 32),
            dropout=kwargs.get("lora_dropout", 0.1),
        )

        lora_config = ExperimentConfig(
            name=f"{task}_lora",
            task=task,
            training_mode="lora",
            **kwargs
        )
        results["lora"] = self.run(
            lora_config,
            lora_model,
            train_loader,
            val_loader,
        )

        # Comparison summary
        print("\n" + "=" * 50)
        print("Comparison Summary")
        print("=" * 50)
        print(f"Full FT: IoU = {results['full_ft'].metrics.get('best_val_loss', 'N/A')}")
        print(f"LoRA:    IoU = {results['lora'].metrics.get('best_val_loss', 'N/A')}")

        return results