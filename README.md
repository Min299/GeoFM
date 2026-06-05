# GeoFM_Minor_Project

A research scaffold for a **Geospatial Foundation Model (GeoFM)**: multitask fine-tuning,
LoRA adaptation, metadata fusion, and knowledge distillation over Earth-observation (EO)
datasets (flood, burn, crop, LULC, NDVI).

## Layout

```
geofm/
├── configs/        # YAML configs for datasets, models, training, experiments
├── datasets/       # Dataset loaders grouped by task (flood, burn, crop, lulc, ndvi)
├── metadata/       # GeoTIFF / timestamp / coordinate parsing + metadata encoding
├── models/         # Backbones, LoRA injection, decoders, multitask heads, fusion
├── losses/         # Segmentation, classification, regression, dice, focal, multitask
├── trainers/       # Training/eval loops (finetune, multitask, distillation)
├── evaluation/     # Metrics and benchmarking per task
├── distillation/   # Feature/multi-level/multi-teacher distillation
├── multitask/      # Task scheduling, registry, batch routing, sampling
├── experiments/    # Per-experiment working dirs (exp01..exp05)
├── deployment/     # Inference, model registry, ONNX export, API, Docker
├── scripts/        # CLI entrypoints: train / evaluate / distill / export / benchmark
├── tests/          # Unit tests
├── notebooks/      # Exploratory notebooks
├── docs/           # Architecture, datasets, experiments, results
└── outputs/        # Checkpoints, logs, metrics, visualizations, reports (gitignored)
```

## Status

Initial scaffold — module files are stubs (`TODO`) ready to be implemented.

## Getting started

```bash
# (optional) create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# install dependencies once a requirements.txt / pyproject is added
# pip install -e .
```
