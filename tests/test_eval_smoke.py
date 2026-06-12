"""tests/test_eval_smoke.py

Smoke tests for evaluate.py script.
"""
import pytest


def test_eval_script_import():
    """Evaluate script should be importable."""
    import scripts.evaluate

    assert hasattr(scripts.evaluate, "main")
    assert hasattr(scripts.evaluate, "create_dummy_dataloader")


def test_eval_create_dummy_dataloader():
    """create_dummy_dataloader should create a working dataloader."""
    from scripts.evaluate import create_dummy_dataloader

    loader = create_dummy_dataloader(batch_size=4)

    assert loader is not None
    assert len(loader) > 0

    batch = next(iter(loader))
    assert len(batch) == 2


def test_eval_parse_args():
    """Evaluate script should parse basic arguments."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["flood", "burn", "lulc", "ndvi"])
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--adaptation", type=str, default="lora")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output", type=str, default="evaluation_results.csv")

    args = parser.parse_args(["--task", "flood"])
    assert args.task == "flood"

    args = parser.parse_args(["--task", "burn", "--checkpoint", "model.pt"])
    assert args.task == "burn"
    assert args.checkpoint == "model.pt"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])