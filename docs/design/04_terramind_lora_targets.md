# TerraMind LoRA Target-Module Report

**Version**: 1.0  
**Date**: 2026-06-05  
**Status**: PENDING VERIFICATION

---

## 1. Purpose

Before implementing LoRA, we must identify the exact module names in TerraMind/TerraTorch where LoRA will be injected.

**WARNING**: Do NOT guess module names. This report requires verification against actual TerraTorch code.

---

## 2. Known Information

### 2.1 TerraTorch Architecture (from docs)

```
TerraMindEncoder
├── tokenizer (modality-specific)
├── embeddings
├── transformer_blocks[]
│   ├── attention
│   │   ├── q_proj
│   │   ├── k_proj
│   │   ├── v_proj
│   │   └── out_proj
│   ├── norm1 (LayerNorm)
│   ├── mlp
│   │   ├── fc1
│   │   ├── act (GELU)
│   │   └── fc2
│   └── norm2 (LayerNorm)
└── norm (final LayerNorm)
```

### 2.2 Feature Extraction Points

| Name | Layer Index | Shape |
|------|-------------|-------|
| F2 | Layer 2 | (B, N, D) |
| F5 | Layer 5 | (B, N, D) |
| F8 | Layer 8 | (B, N, D) |
| F11 | Layer 11 | (B, N, D) |

---

## 3. Required Verification Steps

### 3.1 Step 1: Inspect TerraTorch Code

```bash
# Find terratorch installation
python -c "import terratorch; print(terratorch.__file__)"

# Or check IBM's public repos
# https://github.com/IBM/terratorch
```

### 3.2 Step 2: List Module Names

```python
from terratorch.models import TerraMindFactory

# Create model
model = TerraMindFactory.create({"model_name": "terramind_v1_base"})

# List all modules
for name, module in model.named_modules():
    if 'attention' in name.lower() or 'proj' in name.lower():
        print(f"{name}: {type(module).__name__}")
```

### 3.3 Step 3: Verify Projection Names

Expected patterns:
- `attention.q_proj` or `attn.q_proj` or `q_proj`
- `attention.k_proj` or `attn.k_proj` or `k_proj`
- `attention.v_proj` or `attn.v_proj` or `v_proj`

---

## 4. LoRA Target-Module Options

### 4.1 Option A: Q and V Only (Conservative)

```python
target_modules = ["q_proj", "v_proj"]
```

| Pros | Cons |
|------|------|
| Fewer parameters (~0.5-1%) | May underfit |
| Less overfitting risk | Missing K contribution |
| Standard in many papers | |

**Expected Parameters**: ~17M (base model)

### 4.2 Option B: Q, K, V (Standard)

```python
target_modules = ["q_proj", "k_proj", "v_proj"]
```

| Pros | Cons |
|------|------|
| Balanced | ~35M params |
| Standard LoRA | |
| Well-tested | |

**Expected Parameters**: ~35M (base model)

### 4.3 Option C: Q, K, V, O (Full Attention)

```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

| Pros | Cons |
|------|------|
| Maximum adaptation | ~45M params |
| Best performance potential | Slower |
| | More overfitting risk |

**Expected Parameters**: ~45M (base model)

### 4.4 Option D: Including MLP

```python
target_modules = ["q_proj", "k_proj", "v_proj", "mlp.fc1", "mlp.fc2"]
```

| Pros | Cons |
|------|------|
| Adapts FFN too | ~50M params |
| Maximum capacity | |
| Most parameters | Most compute |

**Expected Parameters**: ~50M (base model)

---

## 5. Recommended Target Modules

Based on standard ViT architecture and GeoFM's research goals:

| Priority | Target Modules | Parameters | Use Case |
|----------|--------------|------------|----------|
| 1 (Recommended) | q_proj, v_proj | ~17M | Initial experiments |
| 2 | q_proj, k_proj, v_proj | ~35M | Standard comparison |
| 3 | q_proj, k_proj, v_proj, o_proj | ~45M | Full adaptation |

**Recommended for Phase 1**: Option 1 (q_proj, v_proj) to minimize parameters while establishing baseline.

---

## 6. LoRA Configuration

### 6.1 Standard Configuration

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                      # Rank
    lora_alpha=32,            # Scaling factor (typically 2×r)
    target_modules=["q_proj", "v_proj"],  # MUST BE VERIFIED
    lora_dropout=0.1,
    bias="none",              # Don't train biases
    task_type="FEATURE_EXTRACTION"  # Or "SEMANTIC_SEGMENTATION"
)
```

### 6.2 Alternative: Manual Injection

```python
class LoRAInjection:
    """Manually inject LoRA without PEFT."""
    
    def __init__(self, rank=16, alpha=32):
        self.rank = rank
        self.alpha = alpha
    
    def inject(self, model, target_modules):
        for name, module in model.named_modules():
            if any(tm in name for tm in target_modules):
                self._inject_lora(module, name)
    
    def _inject_lora(self, layer, name):
        # Replace linear with LoRA linear
        # ...
```

---

## 7. Verification Checklist

- [ ] Find TerraTorch source code
- [ ] List all module names in TerraMindEncoder
- [ ] Verify q_proj, k_proj, v_proj names
- [ ] Check if names have prefixes (e.g., `blocks.0.`)
- [ ] Test LoRA injection on small model
- [ ] Verify trainable parameter count

---

## 8. Module Name Patterns (For Reference)

Based on common ViT implementations:

```python
# Pattern 1: Direct names
"q_proj", "k_proj", "v_proj", "o_proj"

# Pattern 2: With prefix
"attention.q_proj", "attention.k_proj", "attention.v_proj"

# Pattern 3: Block-scoped
"blocks.0.attention.q_proj", "blocks.0.mlp.fc1"

# Pattern 4: TerraTorch-specific (UNKNOWN - needs verification)
"encoder.blocks.0.attn.q_proj"
"encoder.blocks.0.mlp.fc1"
```

---

## 9. Interim Recommendations

Until TerraMind module names are verified:

1. **DO NOT** implement LoRA injection
2. **DO** implement Feature Adapter (which doesn't require module inspection)
3. **DO** prepare the code structure for LoRA injection
4. **DO** document the required `target_modules` as `["TODO: VERIFY"]`

---

## 10. Next Steps

1. **IMMEDIATE**: Inspect TerraTorch source code
2. **THIS WEEK**: Generate verified module name list
3. **NEXT WEEK**: Implement LoRA injection with verified targets
4. **EXPERIMENT**: Compare LoRA vs Feature Adapter

---

## 11. References

- PEFT Library: https://github.com/huggingface/peft
- LoRA Paper: https://arxiv.org/abs/2106.09685
- TerraTorch: https://github.com/IBM/terratorch (when public)