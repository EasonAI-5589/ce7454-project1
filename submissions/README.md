# Submission History

## Submission v1 - F-Score 0.72

**Date**: 2025-10-08
**File**: `submission-v1_f0.72.zip`
**Validation F-Score**: 0.6819
**Test F-Score**: **0.72** ✅

### Model Details
- **Architecture**: MicroSegFormer with LMSA module
- **Checkpoint**: `checkpoints/microsegformer_20251007_153857/best_model.pth`
- **Training Epoch**: 79
- **Parameters**: ~1.75M (within 1.82M limit)

### Training Configuration
```yaml
model:
  name: microsegformer
  use_lmsa: true
  dropout: 0.15

loss:
  ce_weight: 1.0
  dice_weight: 1.0
  use_focal: false
  use_class_weights: false

training:
  learning_rate: 8e-4
  optimizer: AdamW
  scheduler: CosineAnnealingLR
  batch_size: 32
  epochs: 200 (early stopped at 79)
```

### Key Features
1. **LMSA Module**: Lightweight Multi-Scale Attention for small object detection
2. **Dice Loss**: Effective for handling class imbalance (weight=1.0)
3. **Data Augmentation**: Horizontal flip, rotation, color jitter

### Performance Analysis
- **Validation → Test**: 0.6819 → 0.72 (+5.6% improvement!)
- Test set generalization is excellent
- Model performs better on test than validation

### Submission Contents
```
submission-v1_f0.72.zip
├── solution/
│   ├── ckpt.pth (20.2 MB)
│   ├── requirements.txt
│   ├── run.py (LMSA enabled)
│   └── microsegformer.py (with LMSA)
└── masks/ (100 PNG files)
```

---

## Next Steps for v2

Based on v1 results (0.72), potential improvements:

### Option 1: Focal Loss (Target: 0.74-0.76)
- Enable Focal Loss to better handle class imbalance
- Config: `configs/lmsa_focal.yaml`
- Expected gain: +3-5%

### Option 2: Aggressive Optimization (Target: 0.75-0.77)
- Focal Loss + Enhanced Dice weight
- Config: `configs/lmsa_aggressive.yaml`
- Expected gain: +4-7%

### Option 3: Model Ensemble (if allowed)
- Combine multiple models
- Expected gain: +2-3%

**Recommended**: Start training Focal Loss version immediately!
