# SharedModel Architecture Report

**Version**: 1.0  
**Date**: 2026-06-05  
**Status**: DRAFT

---

## 1. Overview

The SharedModel is the core architecture for GeoFM's multi-task geospatial segmentation/regression research.

**Design Goal**: Single frozen TerraMind encoder, task-specific adapter banks, task-specific decoder banks.

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INPUT MODALITIES                              │
│                                                                         │
│   S2L1C (13 bands)    S1GRD (2 bands)    RGB (3 bands)    DEM (1 band)  │
│         │                  │                  │              │        │
│         └──────────────────┴──────────────────┴──────────────┘        │
│                                   ↓                                      │
│                          Tokenizer / Encoder                            │
│                         (TerraMind internal)                            │
│                                   ↓                                      │
│                    ┌──────────────────────────────┐                     │
│                    │    Shared TerraMind Encoder   │                     │
│                    │        (FROZEN)              │                     │
│                    │                              │                     │
│                    │   Output: [F2, F5, F8, F11]   │                     │
│                    │   Shape: (B, N, D) per level  │                     │
│                    └──────────────────────────────┘                     │
│                                   ↓                                      │
│                    ┌──────────────────────────────┐                     │
│                    │      Adapter Bank           │                     │
│                    │  ┌────────────────────────┐  │                     │
│                    │  │  Flood Adapter        │  │                     │
│                    │  │  Burn Adapter          │  │                     │
│                    │  │  LULC Adapter         │  │                     │
│                    │  │  Crop Adapter         │  │                     │
│                    │  │  NDVI Adapter         │  │                     │
│                    │  └────────────────────────┘  │                     │
│                    └──────────────────────────────┘                     │
│                                   ↓                                      │
│                    ┌──────────────────────────────┐                     │
│                    │      Decoder Bank            │                     │
│                    │  ┌────────────────────────┐  │                     │
│                    │  │  Flood Decoder         │  │                     │
│                    │  │  Burn Decoder          │  │                     │
│                    │  │  LULC Decoder          │  │                     │
│                    │  │  Crop Head             │  │                     │
│                    │  │  NDVI Head             │  │                     │
│                    │  └────────────────────────┘  │                     │
│                    └──────────────────────────────┘                     │
│                                   ↓                                      │
│                         TASK-SPECIFIC OUTPUTS                           │
│                                                                         │
│   Flood Mask    Burn Mask    LULC Map    Crop Class    NDVI Map         │
│   (B,2,H,W)     (B,2,H,W)   (B,10,H,W)  (B,9)         (B,1,H,W)        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Components

### 3.1 Shared TerraMind Encoder

| Property | Value |
|----------|-------|
| Status | FROZEN (no gradient) |
| Input | Modality dict (TerraTorch format) |
| Output | List[Tensor] of 4 feature levels |
| Feature dim (D) | 768 (base) or 1024 (large) |
| Tokens (N) | 256 (14×14 + CLS) or 1024 (32×32) |

### 3.2 Adapter Bank

| Task | Adapter Type | Parameters (rank=16) |
|------|-------------|---------------------|
| Flood | Bottleneck Feature Adapter | ~200K |
| Burn | Bottleneck Feature Adapter | ~200K |
| LULC | Bottleneck Feature Adapter | ~200K |
| Crop | Bottleneck Feature Adapter | ~200K |
| NDVI | Bottleneck Feature Adapter | ~200K |
| **Total** | | **~1M** |

### 3.3 Decoder Bank

| Task | Decoder Type | Parameters |
|------|-------------|------------|
| Flood | UNet-style | ~3M |
| Burn | UNet-style | ~3M |
| LULC | UNet-style | ~3M |
| Crop | Classification head | ~100K |
| NDVI | Regression head | ~50K |
| **Total** | | **~9M** |

---

## 4. Parameter Summary

| Component | Trainable | Frozen |
|-----------|-----------|--------|
| TerraMind Encoder | 0 | 7B |
| Adapter Bank (5 tasks) | ~1M | 0 |
| Decoder Bank (5 tasks) | ~9M | 0 |
| **Total** | **~10M** | **7B** |

**Efficiency**: 0.14% of full model

---

## 5. Three Adaptation Strategies

### Strategy A: LoRA Only
```
Frozen TerraMind + LoRA(Q,K,V) → Adapter Bank → Decoder Bank
```
- LoRA modifies attention projections
- No feature-level adapters
- **Expected**: ~35M params (LoRA) + ~9M (decoders)

### Strategy B: Feature Adapter Only
```
Frozen TerraMind → Feature Adapter Bank → Decoder Bank
```
- No attention modification
- Feature-level bottleneck adapters
- **Expected**: ~1M params (adapters) + ~9M (decoders)

### Strategy C: Hybrid (LoRA + Feature Adapter) ⭐
```
Frozen TerraMind + LoRA → Feature Adapter Bank → Decoder Bank
```
- Attention modification + feature adaptation
- **Expected**: ~36M params (LoRA+adapters) + ~9M (decoders)

---

## 6. Task Specifications

| Task | Output | Loss | Metric |
|------|--------|------|--------|
| Flood | (B, 2, H, W) | Dice+CE | IoU |
| Burn | (B, 2, H, W) | Dice+CE | IoU |
| LULC | (B, 10, H, W) | CE | mIoU |
| Crop | (B, 9) | CE | Accuracy |
| NDVI | (B, 1, H, W) | MSE | MAE |

---

## 7. Inference Protocol

```python
class SharedModel:
    def __init__(self, adapter_type: str):  # "lora" | "feature" | "hybrid"
        self.encoder = TerraMindBackbone(pretrained=True, frozen=True)
        self.adapter_bank = AdapterBank(adapter_type)
        self.decoder_bank = DecoderBank()
    
    def forward(self, mod_dict, task: str):
        # 1. Encode
        features = self.encoder(mod_dict)  # [F2, F5, F8, F11]
        
        # 2. Adapt
        adapted = self.adapter_bank(features, task)
        
        # 3. Decode
        output = self.decoder_bank(adapted, task)
        
        return output
    
    def set_task(self, task: str):
        """Switch active adapter + decoder"""
        self.adapter_bank.set_task(task)
        self.decoder_bank.set_task(task)
```

---

## 8. Training Protocol

### Phase 1: Single-Task Training (for each task)
```
For task in [flood, burn, lulc, crop, ndvi]:
    1. Set task adapter + decoder active
    2. Freeze encoder + all other adapters/decoders
    3. Train task adapter + decoder
    4. Save checkpoint
```

### Phase 2: Joint Training (optional)
```
1. Unfreeze all adapters + decoders
2. Train jointly with multi-task loss
3. Use Dynamic Weight Average for loss balancing
```

---

## 9. Open Questions

- [ ] Should adapters share parameters across tasks? (No, keep separate)
- [ ] Should decoders be initialized from ImageNet pretrained? (No, train from scratch)
- [ ] Should we use distillation during single-task training? (No, save for Phase 3)
- [ ] Should we use metadata during Phase 1? (No, add after adaptation is selected)

---

## 10. Implementation Checklist

- [ ] Create `SharedModel` class
- [ ] Create `AdapterBank` with `set_task()` method
- [ ] Create `DecoderBank` with `set_task()` method
- [ ] Implement single-task training loop
- [ ] Implement multi-task evaluation
- [ ] Add checkpoint saving/loading for adapter banks
- [ ] Add inference mode with task switching

---

## 11. References

- AdaptFormer: https://arxiv.org/abs/2205.09535
- LoRA: https://arxiv.org/abs/2106.09685
- TerraTorch: (IBM internal documentation)