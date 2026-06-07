# Decoder Strategy - LOCKED

**Version**: 1.0  
**Date**: 2026-06-05  
**Status**: LOCKED ✅

---

## Decoder Selection by Task

| Task | Decoder Type | Notes |
|------|-------------|-------|
| **Flood** | UNet-style (PyramidDecoder) | Standard segmentation |
| **Burn Scar** | UNet-style (PyramidDecoderWithSkips) | Fine boundaries benefit from skips |
| **LULC** | UNet-style (PyramidDecoder) | Standard segmentation |
| **Crop** | TBD | Depends on dataset (see below) |
| **NDVI** | Regression Head | Lightweight 1×1 Conv style |

---

## Justification

### Flood → UNet-style
- UNet/UNet++/DeepLabV3+ dominate flood segmentation
- TerraMind examples use UNetDecoder for Sen1Floods11
- Simple, strong baseline

### Burn Scar → UNet-style with Skip Connections
- Fine boundary prediction critical for burn scars
- Skip connections help preserve edge information
- Use PyramidDecoderWithSkips

### LULC → UNet-style
- Large number of classes (10+)
- UNet handles multi-class segmentation well
- TerraMind examples consistent

### Crop → TBD (Dataset-dependent)
- **Field-Level Classification**: Use GlobalAvgPool + MLP
- **Pixel-wise Crop Mapping**: Use UNet-style

**ACTION REQUIRED**: Lock crop dataset before implementing decoder

### NDVI → Regression Head
- NOT segmentation
- Use lightweight 1×1 Conv regression head
- No upsampling needed
- Direct mapping from features to NDVI values

---

## What NOT to Ablate

The research question is **adaptation strategy**, not decoder architecture.

- Do NOT compare UNet vs DeepLabV3+ vs SegFormer
- Do NOT tune decoder depth/width
- Use the standard UNet-style decoder for all segmentation tasks
- Focus compute on adapter comparison experiments

---

## Implementation Notes

### Flood/Burn/LULC: Use PyramidDecoder
```python
decoder = PyramidDecoder(
    feature_dim=768,
    num_classes=num_classes,  # 2 for binary, 10 for LULC
)
```

### Burn (Fine Boundaries): Use PyramidDecoderWithSkips
```python
decoder = PyramidDecoderWithSkips(
    feature_dim=768,
    num_classes=2,
)
```

### NDVI: Use Regression Head
```python
decoder = RegressionHead(feature_dim=768)
# Output: (B, 1, H, W) NDVI values
```

### Crop: WAIT until dataset confirmed
```python
# DO NOT implement until dataset is locked
if dataset == "field_classification":
    decoder = ClassificationHead(...)
elif dataset == "pixel_mapping":
    decoder = PyramidDecoder(...)
```

---

## Open Questions

- [x] Flood decoder? → UNet-style ✅
- [x] Burn decoder? → UNet-style with skips ✅
- [x] LULC decoder? → UNet-style ✅
- [ ] Crop decoder? → WAIT for dataset confirmation
- [x] NDVI decoder? → Regression Head ✅