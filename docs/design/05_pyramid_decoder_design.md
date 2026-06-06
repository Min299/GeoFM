# Pyramid Decoder Design Report

**Version**: 1.0  
**Date**: 2026-06-05  
**Status**: DRAFT

---

## 1. Overview

The current decoder only uses the final feature level (F11), discarding 75% of the extracted features. This report designs a proper multi-scale pyramid decoder.

**Problem**: Current implementation uses `features[-1]` only.

**Solution**: Multi-scale decoder using all 4 feature levels [F2, F5, F8, F11].

---

## 2. Feature Levels

| Level | Transformer Layer | Spatial Size | Shape |
|-------|------------------|--------------|-------|
| F2 | Layer 2 | 16×16 | (B, 256, D) |
| F5 | Layer 5 | 16×16 | (B, 256, D) |
| F8 | Layer 8 | 16×16 | (B, 256, D) |
| F11 | Layer 11 | 16×16 | (B, 256, D) |

**Note**: All features have same spatial size (16×16) but represent different semantic levels.

---

## 3. Pyramid Decoder Architecture

### 3.1 High-Level Design

```
Input Features: [F2, F5, F8, F11]
                Each: (B, 256, D) where D=768

                    ┌─────────────────────┐
                    │   Feature Reshaper   │
                    │  (B,256,D)→(B,D,16,16)│
                    └─────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ↓                  ↓                  ↓
       ┌───────┐          ┌───────┐          ┌───────┐
       │ Level │          │ Level │          │ Level │
       │  F2   │          │  F8   │          │ F11   │
       │(fine) │          │       │          │(deep) │
       └───┬───┘          └───┬───┘          └───┬───┘
           │                  │                  │
           │                  │                  │
           └──────────────────┴──────────────────┘
                              │
                    ┌─────────────────────┐
                    │  Multi-Scale Fusion │
                    │   + Skip Conns      │
                    └─────────────────────┘
                              │
                    ┌─────────────────────┐
                    │  Segmentation Head  │
                    │  (B, num_classes,    │
                    │   16, 16)           │
                    └─────────────────────┘
                              │
                    ┌─────────────────────┐
                    │   Upsample to H×W    │
                    │  (B, num_classes,    │
                    │   H, W)              │
                    └─────────────────────┘
```

### 3.2 Alternative: Bottom-Up with Skip Connections

```
F11 ──→ Conv ──→ Up ──→ Conv ──→ Up ──→ Conv ──→ Up ──→ Output
                                     ↑
F8 ──────────────────────────────→ Conv ←─ Skip
                                     ↑
F5 ──────────────────────────────→ Conv ←─ Skip
                                     ↑
F2 ──────────────────────────────→ Conv ←─ Skip
```

---

## 4. Implementation

### 4.1 FeatureReshaper

```python
class FeatureReshaper(nn.Module):
    """Convert (B, N, D) to (B, D, H, W).
    
    Note: Assuming N=256 (16×16 spatial + CLS token).
    In production, should use learned deprojection or positional attention.
    """
    
    def __init__(self, feature_dim: int = 768):
        super().__init__()
        self.feature_dim = feature_dim
        # Skip CLS token (index 0)
        self.skip_cls = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D) where N=256
        B, N, D = x.shape
        
        if self.skip_cls:
            x = x[:, 1:, :]  # Remove CLS token: (B, 255, D)
            N = N - 1
        
        # Reshape to spatial: (B, D, 16, 16)
        # Note: 255 tokens is not a perfect square
        # Fallback to 16×16 with padding or use all 256
        H = W = 16
        
        return x[:, :256, :].view(B, D, H, W)
```

### 4.2 PyramidDecoder

```python
class PyramidDecoder(nn.Module):
    """Multi-scale pyramid decoder using all feature levels."""
    
    def __init__(
        self,
        feature_dim: int = 768,
        num_classes: int = 2,
        channels: tuple = (512, 256, 128, 64),
    ):
        super().__init__()
        
        # Project each feature level to same channel dimension
        self.projections = nn.ModuleList([
            nn.Conv2d(feature_dim, channels[0], 1)
            for _ in range(4)  # 4 feature levels
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(channels[0] * 4, channels[0], 1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        in_ch = channels[0]
        for out_ch in channels[1:]:
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
            in_ch = out_ch
        
        # Upsample layers
        self.upsamples = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            for _ in range(len(channels) - 1)
        ])
        
        # Segmentation head
        self.seg_head = nn.Conv2d(channels[-1], num_classes, 1)
        
        # Final upsample to match input size
        self.final_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of 4 tensors [F2, F5, F8, F11]
                     Each shape: (B, N, D) = (B, 256, 768)
        
        Returns:
            Segmentation logits: (B, num_classes, H, W)
        """
        # Reshape and project each feature level
        reshaped = []
        for feat, proj in zip(features, self.projections):
            # (B, N, D) → (B, D, 16, 16)
            B, N, D = feat.shape
            H = W = 16
            spatial = feat[:, :256, :].view(B, D, H, W)
            
            # Project to decoder channels
            projected = proj(spatial)
            reshaped.append(projected)
        
        # Concatenate all feature levels
        # Each is (B, channels[0], 16, 16)
        fused = torch.cat(reshaped, dim=1)  # (B, channels[0] * 4, 16, 16)
        
        # Fuse multi-scale features
        fused = self.fusion(fused)  # (B, channels[0], 16, 16)
        
        # Decode
        x = fused
        for i, (block, upsample) in enumerate(zip(self.decoder_blocks, self.upsamples)):
            x = upsample(x)
            x = block(x)
        
        # Segment
        out = self.seg_head(x)  # (B, num_classes, 64, 64)
        out = self.final_upsample(out)  # (B, num_classes, 256, 256)
        
        return out
```

---

## 5. Skip Connection Variant

### 5.1 With Skip Connections

```python
class PyramidDecoderWithSkips(nn.Module):
    """Pyramid decoder with skip connections from each level."""
    
    def __init__(self, feature_dim=768, num_classes=2, channels=(512, 256, 128, 64)):
        super().__init__()
        
        # Projections for each level
        self.projections = nn.ModuleList([
            nn.Conv2d(feature_dim, channels[i], 1)
            for i in range(4)
        ])
        
        # Decoder blocks with skip inputs
        # Level 3 (F11) → deepest
        self.decoder_3 = DecoderBlock(channels[3], 0, channels[3])
        
        # Level 2 (F8) → skip to decoder 1
        self.decoder_2 = DecoderBlock(channels[2], channels[2], channels[2])
        
        # Level 1 (F5) → skip to decoder 0
        self.decoder_1 = DecoderBlock(channels[1], channels[1], channels[1])
        
        # Level 0 (F2) → final
        self.decoder_0 = DecoderBlock(channels[0], channels[0], channels[0])
        
        # Segmentation head
        self.seg_head = nn.Conv2d(channels[0], num_classes, 1)
    
    def forward(self, features):
        # Reshape all features
        spatial = [self._reshape(f) for f in features]
        
        # Project to decoder channels
        proj = [proj(s) for proj, s in zip(self.projections, spatial)]
        
        # Decode from deepest to shallowest
        # F11 (16×16) → F8 (32×32) → F5 (64×64) → F2 (128×128)
        x = self.decoder_3(proj[3])  # F11 only
        
        x = self.decoder_2(x, proj[2])  # F11 + F8
        x = self.decoder_1(x, proj[1])  # + F5
        x = self.decoder_0(x, proj[0])  # + F2
        
        return self.seg_head(x)
```

---

## 6. Parameter Count

| Component | Parameters |
|-----------|------------|
| Projections (4 × Conv2d 768→512) | 4 × 768 × 512 = 1.5M |
| Fusion (Conv2d 2048→512) | 2M |
| Decoder blocks (3 blocks) | ~2M |
| Segmentation head | ~1K |
| **Total** | **~5.5M** |

---

## 7. Comparison: Current vs Pyramid

| Aspect | Current (Last Feature Only) | Pyramid (Multi-Scale) |
|--------|------------------------------|----------------------|
| Features used | F11 only | F2, F5, F8, F11 |
| Multi-scale | No | Yes |
| Skip connections | No | Yes |
| Parameters | ~2M | ~5.5M |
| Expected IoU | Baseline | +5-10% |
| Complexity | Simple | Moderate |

---

## 8. Implementation Notes

### 8.1 CLS Token Handling
- Skip CLS token (index 0)
- Use 256 tokens for 16×16 spatial grid

### 8.2 Spatial Alignment
- All features have same spatial size (16×16)
- No upsampling needed before concatenation
- Simple concatenation is sufficient

### 8.3 Performance Considerations
- 256 tokens = 16×16 spatial is coarse
- For higher resolution output, upsample progressively
- Consider using position embeddings for better spatial awareness

---

## 9. Testing Checklist

- [ ] Test with dummy features (B, 256, 768)
- [ ] Verify all 4 levels are used
- [ ] Check output shape matches input
- [ ] Verify gradients flow to all levels
- [ ] Compare with current single-level decoder

---

## 10. Open Questions

- [ ] Should we use attention for multi-scale fusion? (No, keep simple)
- [ ] Should we upsample features before concatenation? (No, same size)
- [ ] Should we use FPN instead? (No, simple concat is fine for 4 levels)
- [ ] Should output be exact input resolution? (Yes, use interpolation)

---

## 11. References

- FPN: https://arxiv.org/abs/1612.03144
- U-Net: https://arxiv.org/abs/1505.04597
- ViT-FC (feature deprojection): https://arxiv.org/abs/2104.05707