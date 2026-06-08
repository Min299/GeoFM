#!/usr/bin/env python3
"""geofm.scripts.train

Training script for GeoFM experiments.

Usage:
    python scripts/train.py --config configs/experiments/exp01_feature_adapter.yaml
    python scripts/train.py --name exp01 --task flood --adaptation lora --backbone terramind_base
"""
import argparse
import logging
import sys
from pathlib import Path

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from geofm.experiments import ExperimentConfig, ExperimentRunner
from geofm.experiments.experiment_builder import ExperimentBuilder
from geofm.trainers import SharedTrainer, CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_dummy_dataloader(batch_size: int = 8):
    """Create a dummy dataloader for testing.

    In production, replace with actual dataset.
    """
    from torch.utils.data import DataLoader, TensorDataset

    # Dummy data: (image, target) pairs
    images = torch.randn(batch_size, 12, 224, 224)  # S2L2A 12-band
    targets = torch.randint(0, 2, (batch_size, 224, 224))  # Binary mask

    dataset = TensorDataset(images, targets)

    return DataLoader(dataset, batch_size=batch_size)


def main():
    parser = argparse.ArgumentParser(description="Train GeoFM model")

    # Config file option
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file",
    )

    # Or manual options
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--task", type=str, default="flood", help="Task name")
    parser.add_argument("--adaptation", type=str, default="lora", help="Adaptation method")
    parser.add_argument("--backbone", type=str, default="terramind_base", help="Backbone model")

    # Training options
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # LoRA options
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # Paths
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--resume", type=str, help="Checkpoint to resume from")

    args = parser.parse_args()

    # Load or create config
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = ExperimentConfig.from_yaml(args.config)
    else:
        logger.info("Creating config from arguments")
        config = ExperimentConfig(
            name=args.name or f"exp_{args.adaptation}_{args.task}",
            task=args.task,
            adaptation=args.adaptation,
            backbone=args.backbone,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            epochs=args.epochs,
            seed=args.seed,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            output_dir=args.output_dir,
        )

    logger.info(f"Experiment: {config.name}")
    logger.info(f"  Task: {config.task}")
    logger.info(f"  Adaptation: {config.adaptation}")
    logger.info(f"  Backbone: {config.backbone}")
    logger.info(f"  Epochs: {config.epochs}")

    # Set seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # Build model
    logger.info("Building model...")
    model = ExperimentBuilder.build(config)
    logger.info(f"Model built. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create dataloaders (dummy for now)
    train_loader = create_dummy_dataloader(config.batch_size)
    val_loader = create_dummy_dataloader(config.batch_size)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Create trainer
    trainer = SharedTrainer(
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager()

    # Create runner
    runner = ExperimentRunner(
        config=config,
        trainer=trainer,
        checkpoint_manager=checkpoint_manager,
    )

    # Resume if requested
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint_manager.load(args.resume, model, optimizer)

    # Run training
    logger.info("Starting training...")
    trained_model = runner.run()

    # Save final model
    output_path = Path(config.output_dir) / f"{config.name}_final.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_manager.save(str(output_path), trained_model, optimizer)
    logger.info(f"Model saved to {output_path}")

    # Print history summary
    history = runner.get_history()
    if history:
        best_loss = min(h.get("loss", float("inf")) for h in history)
        logger.info(f"Training complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()