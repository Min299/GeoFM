# GeoFM Design Reports

This directory contains architectural design reports for the GeoFM project.

## Reports

| # | Report | Status | Priority |
|---|--------|--------|----------|
| 01 | [SharedModel Architecture](01_shared_model_architecture.md) | DRAFT | 1 |
| 02 | [AdapterBank Architecture](02_adapter_bank_architecture.md) | DRAFT | 1 |
| 03 | [DecoderBank Architecture](03_decoder_bank_architecture.md) | DRAFT | 1 |
| 04 | [TerraMind LoRA Targets](04_terramind_lora_targets.md) | PENDING | 2 |
| 05 | [Pyramid Decoder Design](05_pyramid_decoder_design.md) | DRAFT | 1 |

---

## Quick Summary

### Architecture Overview

```
Shared TerraMind Encoder (FROZEN)
         ↓
    Adapter Bank
  (LoRA / Feature / Hybrid)
         ↓
    Decoder Bank
(UNet / Classification / Regression)
         ↓
    Task Outputs
```

### Experiments

| Exp | Adapter Type | Description |
|-----|-------------|-------------|
| 1A | Feature Adapter Only | Baseline PEFT |
| 1B | LoRA Only | Attention adaptation |
| 1C | Hybrid (LoRA + Feature) | ⭐ Best expected |

### Immediate Actions

1. ✅ Rename `TaskLoRAAdapter` → `TaskFeatureAdapter`
2. 🔄 Create PyramidDecoder using all 4 features
3. 🔄 Create AdapterBank with proper adapters
4. 🔄 Create DecoderBank with task-specific decoders
5. ⏳ Verify TerraMind LoRA targets (after integration)

---

## Review Required

These reports need review before implementation begins:

- [ ] SharedModel architecture (report 01)
- [ ] AdapterBank architecture (report 02)  
- [ ] DecoderBank architecture (report 03)
- [ ] TerraMind LoRA targets (report 04)
- [ ] Pyramid decoder design (report 05)

**Do NOT implement until reports are reviewed.**