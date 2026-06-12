#!/usr/bin/env python3
"""scripts/inspect_terramind.py

Inspect TerraMind backbone to discover:
- Actual module names
- Attention layer structure
- Parameter names for LoRA targeting
- Feature extraction path
"""
import torch
from pprint import pprint

from geofm.models.backbones import build_backbone


def main():
    print("=" * 70)
    print("  TERRAINSPECT - TerraMind Architecture Inspector")
    print("=" * 70)

    # Load model
    print("\n[1] Loading TerraMind Base...")
    model = build_backbone("terramind_base")
    print(f"    Model type: {type(model)}")

    # Parameters summary
    print("\n[2] PARAMETER SUMMARY")
    print("-" * 70)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Total params:     {total:,}")
    print(f"  Trainable:        {trainable:,}")
    print(f"  Frozen:           {frozen:,}")

    # Model structure
    print("\n[3] MODEL STRUCTURE (named_modules)")
    print("-" * 70)
    for name, module in model.named_modules():
        # Print module name and type, but skip deeply nested ones
        depth = name.count('.')
        if depth <= 3:  # Only show top 3 levels
            print(f"  {name:<50} {type(module).__name__}")

    # Detailed parameter inspection
    print("\n[4] PARAMETER NAMES (named_parameters)")
    print("-" * 70)
    for name, param in model.named_parameters():
        if 'attn' in name.lower() or 'qkv' in name.lower() or 'query' in name.lower() or 'key' in name.lower() or 'value' in name.lower():
            print(f"  {name:<60} {str(list(param.shape)):<15} {param.numel():>10,}")

    # Look for attention-related parameters
    print("\n[5] ATTENTION LAYER ANALYSIS")
    print("-" * 70)
    attention_params = {}
    for name, param in model.named_parameters():
        # Check if this is an attention-related parameter
        if any(x in name.lower() for x in ['attn', 'qkv', 'query', 'key', 'value', 'proj']):
            # Extract the layer name
            parts = name.split('.')
            layer_name = '.'.join(parts[:4]) if len(parts) >= 4 else name
            if layer_name not in attention_params:
                attention_params[layer_name] = []
            attention_params[layer_name].append(parts[-1])

    for layer, params in attention_params.items():
        print(f"  {layer:<40} -> {params}")

    # Transformer block structure
    print("\n[6] TRANSFORMER BLOCKS")
    print("-" * 70)
    if hasattr(model, '_model'):
        inner_model = model._model
        if hasattr(inner_model, 'encoder'):
            encoder = inner_model.encoder
            print(f"  Encoder type: {type(encoder).__name__}")
            print(f"  Number of blocks: {len(encoder)}")
            if len(encoder) > 0:
                print(f"\n  First block structure:")
                first_block = encoder[0]
                for name, param in first_block.named_parameters():
                    print(f"    {name:<40} {str(list(param.shape))}")

    # Feature extraction path
    print("\n[7] FEATURE EXTRACTION PATH")
    print("-" * 70)
    if hasattr(model, '_model'):
        inner_model = model._model
        print(f"  out_channels: {inner_model.out_channels}")
        print(f"  modalities: {inner_model.modalities}")

    # Output format
    print("\n[8] OUTPUT FORMAT")
    print("-" * 70)
    print(f"  feature_indices: {model.feature_indices}")
    print(f"  _feature_dim: {model._feature_dim}")
    print(f"  _num_layers: {model._num_layers}")

    # Freeze test
    print("\n[9] FREEZE TEST")
    print("-" * 70)
    model.freeze()
    frozen_after = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  After freeze(): {frozen_after:,} params frozen")
    print(f"  is_frozen(): {model.is_frozen()}")

    # Final summary
    print("\n" + "=" * 70)
    print("  SUMMARY FOR LoRA TARGETING")
    print("=" * 70)

    # Find all qkv-related parameters
    qkv_params = []
    for name, param in model.named_parameters():
        if any(x in name.lower() for x in ['qkv', 'query', 'key', 'value']):
            qkv_params.append((name, param.shape, param.numel()))

    if qkv_params:
        print("\n  QKV-Related Parameters (LoRA targets):")
        for name, shape, numel in qkv_params[:20]:  # Show first 20
            print(f"    {name:<60} {str(list(shape)):<15} {numel:>10,}")
    else:
        print("\n  No QKV parameters found - checking alternative names...")
        for name, param in model.named_parameters():
            if 'attn' in name.lower():
                print(f"    {name:<60} {str(list(param.shape)):<15} {param.numel():>10,}")


if __name__ == "__main__":
    main()