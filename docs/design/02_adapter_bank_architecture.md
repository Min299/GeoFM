# AdapterBank Architecture Report

**Version**: 1.0  
**Date**: 2026-06-05  
**Status**: DRAFT

---

## 1. Overview

AdapterBank manages task-specific adaptation modules that transform TerraMind's frozen features into task-optimized representations.

**Key Distinction**:
- **LoRA Adapter**: Modifies attention Q,K,V projections (INSIDE backbone)
- **Feature Adapter**: Modifies extracted features (OUTSIDE backbone)
- **Hybrid Adapter**: Combines both

---

## 2. Three Adapter Types

### 2.1 Feature Adapter (Bottleneck Adapter)

```
Input:  (B, N, D) - Feature tensor from TerraMind layer
Output: (B, N, D) - Adapted feature tensor

Architecture:
    x ──────────────────────────────────→ + ──→ output
      │                                   ↑
      ↓                                   │
    Linear(D → r)                         │
      ↓                                   │
    GELU()                                │
      ↓                                   │
    Linear(r → D) ────────────────────────┘

Parameters:
    - D: feature dimension (768 for base, 1024 for large)
    - r: bottleneck rank (typically 16 or 32)
    - Total: 2 × D × r = 2 × 768 × 16 = 24,576 per feature level
```

**Properties**:
- Applied AFTER feature extraction
- Operates on spatial tokens (B, N, D)
- One adapter per feature level (F2, F5, F8, F11)
- 4 adapters per task

### 2.2 LoRA Adapter (Attention Adapter)

```
Applied INSIDE attention layers of TerraMind.

Q_new = Q + (B_Q @ A_Q) @ x / scaling
K_new = K + (B_K @ A_K) @ x / scaling  
V_new = V + (B_V @ A_V) @ x / scaling

Parameters:
    - 3 projection pairs (Q, K, V)
    - Each: 2 × D × r
    - For 12 layers: 12 × 3 × 2 × D × r = 12 × 3 × 2 × 768 × 16 = ~8.9M
```

**Properties**:
- Modifies attention mechanism
- Applied INSIDE backbone
- Requires module-level injection
- Target: q_proj, k_proj, v_proj

### 2.3 Hybrid Adapter

```
Combines LoRA (attention) + Feature Adapter (output)

    ┌─────────────────────────────────────┐
    │  TerraMind + LoRA (Q,K,V)          │
    │           ↓                        │
    │  [F2, F5, F8, F11]                │
    │           ↓                        │
    │  Feature Adapter Bank (per task)   │
    │           ↓                        │
    │  Adapted Features                  │
    └─────────────────────────────────────┘
```

---

## 3. AdapterBank Structure

```python
class AdapterBank(nn.Module):
    """Manages task-specific adapters."""
    
    def __init__(
        self,
        tasks: List[str],           # ["flood", "burn", "lulc", "crop", "ndvi"]
        adapter_type: str,         # "lora" | "feature" | "hybrid"
        feature_dim: int = 768,
        rank: int = 16,
        num_levels: int = 4,       # F2, F5, F8, F11
    ):
        # Create adapters for each task
        self._task_adapters = nn.ModuleDict()
        
        for task in tasks:
            if adapter_type == "feature":
                self._task_adapters[task] = FeatureAdapterBank(
                    feature_dim=feature_dim,
                    rank=rank,
                    num_levels=num_levels,
                )
            elif adapter_type == "lora":
                # LoRA needs backbone reference - handled separately
                self._task_adapters[task] = None
            elif adapter_type == "hybrid":
                self._task_adapters[task] = HybridAdapterBank(...)
        
        self._current_task = None
    
    def set_task(self, task: str) -> None:
        """Activate adapters for a specific task."""
        if task not in self._task_adapters:
            raise ValueError(f"Unknown task: {task}")
        self._current_task = task
    
    def forward(self, features: List[Tensor]) -> List[Tensor]:
        """Apply current task's adapters to features."""
        if self._current_task is None:
            raise RuntimeError("No task set. Call set_task() first.")
        
        adapter = self._task_adapters[self._current_task]
        return adapter(features)
```

---

## 4. FeatureAdapterBank Implementation

```python
class FeatureAdapterBank(nn.Module):
    """4 bottleneck adapters, one per feature level."""
    
    def __init__(self, feature_dim: int, rank: int, num_levels: int = 4):
        super().__init__()
        self.adapters = nn.ModuleList([
            BottleneckAdapter(feature_dim, rank)
            for _ in range(num_levels)
        ])
    
    def forward(self, features: List[Tensor]) -> List[Tensor]:
        """Apply adapters to each feature level.
        
        Args:
            features: List of 4 tensors [F2, F5, F8, F11]
                     Each shape: (B, N, D)
        Returns:
            List of 4 adapted tensors
        """
        return [adapter(f) for adapter, f in zip(self.adapters, features)]


class BottleneckAdapter(nn.Module):
    """Standard bottleneck adapter with residual connection."""
    
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(rank, dim, bias=False)
        
        # Initialize
        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)
    
    def forward(self, x: Tensor) -> Tensor:
        return x + self.up(self.act(self.down(x)))
```

---

## 5. Parameter Counts

| Adapter Type | Per Level | Per Task (4 levels) | 5 Tasks |
|-------------|-----------|---------------------|---------|
| Feature Adapter (r=16) | 24,576 | 98,304 | 491,520 |
| Feature Adapter (r=32) | 49,152 | 196,608 | 983,040 |
| LoRA (r=16) | ~743,000 | ~2.97M | 14.85M |
| Hybrid (r=16) | ~768,000 | ~3.07M | 15.34M |

**Note**: LoRA parameters depend on number of attention layers (12 for base model)

---

## 6. Task Adapter Storage

```python
# Each task has its own adapter weights
task_adapters = {
    "flood": FeatureAdapterBank(...),  # 98K params
    "burn": FeatureAdapterBank(...),   # 98K params
    "lulc": FeatureAdapterBank(...),   # 98K params
    "crop": FeatureAdapterBank(...),   # 98K params
    "ndvi": FeatureAdapterBank(...),   # 98K params
}

# Checkpoint format
checkpoint = {
    "flood": {"adapters": [...], "decoder": [...]},
    "burn": {"adapters": [...], "decoder": [...]},
    ...
}
```

---

## 7. LoRA vs Feature Adapter Comparison

| Aspect | LoRA | Feature Adapter |
|--------|------|----------------|
| Location | Inside attention | Outside backbone |
| What it adapts | Attention dynamics | Feature representation |
| Spatial awareness | Indirect | Direct |
| Parameters | ~35M (base) | ~100K (per task) |
| Training speed | Faster | Slower |
| Memory | Higher | Lower |
| Best for | Language tasks | Vision/spatial tasks |

**For GeoFM tasks** (dense spatial prediction):
- Feature Adapter is likely better
- Hybrid combines both

---

## 8. Implementation Checklist

- [ ] Rename `TaskLoRAAdapter` → `TaskFeatureAdapter`
- [ ] Create `BottleneckAdapter` class with GELU
- [ ] Create `FeatureAdapterBank` (4 adapters per task)
- [ ] Create `AdapterBank` manager
- [ ] Add `set_task()` method
- [ ] Update `FloodModel` to use new naming
- [ ] Test adapter switching

---

## 9. Open Questions

- [ ] Should we use LayerNorm before/after adapter? (No, keep simple)
- [ ] Should adapters share initialization? (No, random init is fine)
- [ ] Should we use adapter fusion instead of separate adapters? (No, for interpretability)
- [ ] Should rank be task-specific? (No, use same rank for all)

---

## 10. References

- AdaptFormer: https://arxiv.org/abs/2205.09535
- Bottleneck Adapters: https://arxiv.org/abs/1909.05778
- LoRA: https://arxiv.org/abs/2106.09685