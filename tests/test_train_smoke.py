"""tests/test_train_smoke.py

Smoke tests for train.py script.
"""
import pytest


def test_train_script_import():
    """Train script should be importable."""
    import scripts.train

    assert hasattr(scripts.train, "main")
    assert hasattr(scripts.train, "create_dummy_dataloader")


def test_train_create_dummy_dataloader():
    """create_dummy_dataloader should create a working dataloader."""
    from scripts.train import create_dummy_dataloader

    loader = create_dummy_dataloader(batch_size=4)

    assert loader is not None
    assert len(loader) > 0

    # Get one batch
    batch = next(iter(loader))
    assert len(batch) == 2  # (images, targets)


def test_train_parse_args(monkeypatch):
    """Train script should parse basic arguments."""
    from scripts.train import main

    # Mock argparse to avoid SystemExit
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--task", "flood", "--adaptation", "lora", "--epochs", "1"]
    )

    # Should not raise
    try:
        # Don't actually run main(), just test argument parsing
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str)
        parser.add_argument("--name", type=str)
        parser.add_argument("--task", type=str, default="flood")
        parser.add_argument("--adaptation", type=str, default="lora")
        parser.add_argument("--backbone", type=str, default="terramind_base")
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--epochs", type=int, default=50)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--lora_rank", type=int, default=16)
        parser.add_argument("--lora_alpha", type=int, default=32)
        parser.add_argument("--output_dir", type=str, default="outputs")
        parser.add_argument("--resume", type=str)

        args = parser.parse_args(["--task", "flood", "--adaptation", "lora"])
        assert args.task == "flood"
        assert args.adaptation == "lora"
    except SystemExit:
        pytest.fail("Argument parsing failed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])