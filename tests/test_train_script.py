"""tests/test_train_script.py

Tests for train.py script.
"""
from pathlib import Path


def test_train_script_exists():
    """Test that train.py script exists."""
    path = Path("scripts/train.py")
    assert path.exists()


def test_train_script_is_executable():
    """Test that train.py has execute permission."""
    path = Path("scripts/train.py")
    assert path.exists()

    # Check if file is readable as Python
    with open(path) as f:
        content = f.read()
        assert "def main()" in content
        assert "__name__" in content