#!/usr/bin/env python3
"""scripts/ablation.py

Ablation study script for GeoFM.

Usage:
    python scripts/ablation.py --config configs/ablation.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from geofm.config.config_loader import ConfigLoader


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run GeoFM ablation study")

    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file",
    )

    args = parser.parse_args()

    cfg = ConfigLoader.load(args.config)

    print("Ablation started:")
    print(cfg)


if __name__ == "__main__":
    main()