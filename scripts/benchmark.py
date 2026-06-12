#!/usr/bin/env python3
"""scripts/benchmark.py

Benchmark script for GeoFM.

Usage:
    python scripts/benchmark.py --config configs/benchmark.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from geofm.config.config_loader import ConfigLoader


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark GeoFM model")

    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file",
    )

    args = parser.parse_args()

    cfg = ConfigLoader.load(args.config)

    print("Benchmark started:")
    print(cfg)


if __name__ == "__main__":
    main()