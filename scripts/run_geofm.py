#!/usr/bin/env python
"""scripts/run_geofm.py

GeoFM main entrypoint for training and inference.
"""
from __future__ import annotations

from geofm.config.config_loader import (
    ConfigLoader,
)

from geofm.integration.pipeline import (
    GeoFMPipeline,
)


def main():
    """Main entrypoint for GeoFM."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GeoFM - Geospatial Foundation Model"
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to configuration file",
    )

    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "eval", "predict"],
        help="Running mode",
    )

    args = parser.parse_args()

    cfg = ConfigLoader.load(
        args.config
    )

    pipeline = GeoFMPipeline(cfg)

    print(
        "GeoFM initialized."
    )

    return pipeline


if __name__ == "__main__":

    main()