# DecoderBank Architecture Report

**Version**: 1.0  
**Date**: 2026-06-05  
**Status**: DRAFT

---

## 1. Overview

DecoderBank manages task-specific decoders/heads that convert adapted features into task-specific outputs.

**Design Philosophy**: Keep decoders simple and task-specific. Decoder research is NOT the focus of this project.

---

## 2. Decoder Types by Task

| Task | Output Shape | Decoder Type | Loss Function |
|------|-------------|--------------|---------------|
| Flood | (B, 2, H, W) | UNet-style | Dice + CE |
| Burn Scar | (B, 2, H, W) | UNet-style | Dice + CE |
| LULC | (B, 10, H, W) | UNet-style | Cross-Entropy |
| Crop | (B, 9) | Classification Head | Cross-Entropy |
| NDVI | (B, 1, H, W) | Regression Head | MSE |

---

## 3. UNet-Style Segmentation Decoder

### 3.1 Architecture

```
Adapted Features: [F2, F5, F8, F11]
                   ↓     ↓     ↓     ↓
                 Reshaped to multi-scale spatial

Multi-Scale Fusion:
    F11 (16×16) ──→ Decoder Block ──→ 32×32 ──→ 64×64 ──→ 128×128
    F8  (32×32) ────────────────────────────→ Skip ──→ 128×128
    F5  (64×64) ──────────────────────────────────→ Skip ──→ 128×128
    F2  (128×128)───────────────────────────────────────→ Skip → 128×128

Output: (B, num_classes, 128, 128) → Upsample to (B, num_classes, H, W)
```

### 3.2 Decoder Block

```python
class DecoderBlock(nn.Module):
    """Standard UNet decoder block."""
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None:
            # Align spatial dimensions
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x
```

### 3.3 Feature Reshaping

```python
class FeatureReshaper(nn.Module):
    """Convert (B, N, D) tokens to (B, C, H, W) spatial.
    
    Note: This is a naive reshape. For production, consider:
    - Learned deprojection (ViT-FC)
    - Position embeddings
    - Spatial attention
    """
    
    def __init__(self, feature_dim, num_tokens):
        super().__init__()
        # Assume square spatial
        self.spatial_size = int(num_tokens ** 0.5)
        self.feature_dim = feature_dim
    
    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape
        H = W = self.spatial_size
        return x.view(B, D, H, W)
```

---

## 4. Classification Head (Crop)

### 4.1 Architecture

```python
class ClassificationHead(nn.Module):
    """Image-level classification head."""
    
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        # Global average pooling + classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(feature_dim, num_classes)
    
    def forward(self, features):
        # Use last feature level [F11]
        x = features[-1]  # (B, N, D)
        
        # Reshape to spatial
        B, N, D = x.shape
        H = W = int(N ** 0.5)
        x = x.view(B, D, H, W)
        
        # Global pooling
        x = self.pool(x).view(B, D)
        
        # Classify
        return self.fc(x)  # (B, num_classes)
```

### 4.2 Alternative: Patch-wise Classification

If crop dataset is pixel-wise segmentation instead of image classification:

```python
class SegmentationHead(nn.Module):
    """Pixel-wise classification."""
    
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, num_classes, 1)
        )
    
    def forward(self, features):
        x = features[-1]  # Use last feature
        B, N, D = x.shape
        H = W = int(N ** 0.5)
        x = x.view(B, D, H, W)
        return self.conv(x)  # (B, num_classes, H, W)
```

---

## 5. Regression Head (NDVI)

### 5.1 Architecture

```python
class RegressionHead(nn.Module):
    """NDVI regression head."""
    
    def __init__(self, feature_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, feature_dim // 4, 3, padding=1),
            nn.BatchNorm2d(feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 4, 1, 1)  # Single NDVI value per pixel
        )
    
    def forward(self, features):
        x = features[-1]
        B, N, D = x.shape
        H = W = int(N ** 0.5)
        x = x.view(B, D, H, W)
        return self.conv(x)  # (B, 1, H, W)
```

### 5.2 Output Range

NDVI values are in [-1, 1]. Consider:
- Sigmoid for [0, 1] output
- Tanh for [-1, 1] output
- No activation + MSE = unbounded (not recommended)

```python
# Recommended: Tanh for [-1, 1] NDVI range
self.output_activation = nn.Tanh()
```

---

## 6. DecoderBank Implementation

```python
class DecoderBank(nn.Module):
    """Manages task-specific decoders."""
    
    def __init__(self, tasks: List[str], feature_dim: int = 768):
        super().__init__()
        self._decoders = nn.ModuleDict()
        
        for task in tasks:
            if task in ["flood", "burn", "lulc"]:
                self._decoders[task] = SegmentationDecoder(feature_dim)
            elif task == "crop":
                self._decoders[task] = ClassificationHead(feature_dim, num_classes=9)
            elif task == "ndvi":
                self._decoders[task] = RegressionHead(feature_dim)
        
        self._current_task = None
    
    def set_task(self, task: str) -> None:
        """Activate decoder for a specific task."""
        if task not in self._decoders:
            raise ValueError(f"Unknown task: {task}")
        self._current_task = task
    
    def forward(self, features: List[Tensor]) -> Tensor:
        """Decode adapted features for current task."""
        if self._current_task is None:
            raise RuntimeError("No task set. Call set_task() first.")
        
        decoder = self._decoders[self._current_task]
        return decoder(features)
```

---

## 7. Parameter Counts

| Decoder Type | Parameters |
|-------------|------------|
| UNet-style (4 levels) | ~3M |
| Classification Head | ~100K |
| Regression Head | ~50K |

**Total for 5 tasks**: ~9.2M

---

## 8. Segmentation Decoder Details

### 8.1 Full Implementation

```python
class SegmentationDecoder(nn.Module):
    """UNet-style decoder with multi-scale features."""
    
    def __init__(self, feature_dim=768, num_classes=2, channels=(512, 256, 128, 64)):
        super().__init__()
        
        # Feature reshaping for each level
        self.reshape_layers = nn.ModuleList([
            FeatureReshaper(feature_dim, 256)  # 16×16
            for _ in range(4)
        ])
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(feature_dim, channels[0], 1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        
        # Decoder blocks (bottom-up)
        self.decoder_blocks = nn.ModuleList()
        in_ch = channels[0]
        for out_ch in channels[1:]:
            self.decoder_blocks.append(
                DecoderBlock(in_ch, out_ch, out_ch)
            )
            in_ch = out_ch
        
        # Segmentation head
        self.seg_head = nn.Conv2d(channels[-1], num_classes, 1)
        
        # Upsample
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
    
    def forward(self, features: List[Tensor]) -> Tensor:
        # features: [F2, F5, F8, F11] = [256, 256, 256, 256] tokens
        
        # Reshape all features to spatial
        spatial = [reshape(f) for reshape, f in zip(self.reshape_layers, features)]
        # spatial: [(B,D,16,16), (B,D,16,16), (B,D,16,16), (B,D,16,16)]
        
        # Start with deepest features
        x = self.input_proj(spatial[-1])  # (B, channels[0], 16, 16)
        
        # Decode with skip connections (use same spatial for all)
        for i, block in enumerate(self.decoder_blocks):
            x = block(x)  # Upsample + conv
        
        # Segment
        out = self.seg_head(x)  # (B, num_classes, 64, 64)
        
        # Upsample to original size
        out = self.upsample(out)  # (B, num_classes, 256, 256)
        
        return out
```

---

## 9. Implementation Checklist

- [ ] Create `SegmentationDecoder` class
- [ ] Create `ClassificationHead` class  
- [ ] Create `RegressionHead` class
- [ ] Create `DecoderBank` manager
- [ ] Test multi-scale fusion
- [ ] Verify output shapes

---

## 10. Open Questions

- [ ] Should we use FPN (Feature Pyramid Network) instead of U-Net? (No, U-Net is sufficient)
- [ ] Should decoder be deeper or wider? (Deeper is better for complex tasks)
- [ ] Should we use attention in decoder? (No, keep simple for baseline)
- [ ] Should output be resized to exact input size? (Yes, for segmentation)

---

## 11. References

- U-Net: https://arxiv.org/abs/1505.04597
- Feature Pyramid Network: https://arxiv.org/abs/1612.03144
- SegFormer: https://arxiv.org/abs/2105.15203