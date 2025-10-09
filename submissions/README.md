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

## Submission v2 - F-Score 0.6889 (Val)

**Date**: 2025-10-09
**File**: `submission_v2_f0.6889_codabench.zip`
**Validation F-Score**: 0.6889
**Test F-Score**: **TBD** (Expected: 0.73-0.74)

### Model Details
- **Architecture**: MicroSegFormer with LMSA module
- **Checkpoint**: `checkpoints/microsegformer_20251008_025917/best_model.pth`
- **Training Epoch**: 92
- **Parameters**: ~1.75M (within 1.82M limit)

### Training Configuration
```yaml
model:
  name: microsegformer
  use_lmsa: true
  dropout: 0.15

loss:
  ce_weight: 1.0
  dice_weight: 1.5  # ⬆️ KEY CHANGE: Increased from 1.0
  use_focal: false
  use_class_weights: false

training:
  learning_rate: 8e-4
  optimizer: AdamW
  scheduler: CosineAnnealingLR
  batch_size: 32
  epochs: 200 (early stopped at 92)
```

### Key Improvements over v1
1. **Enhanced Dice Weight**: 1.0 → 1.5 (better small object segmentation)
2. **Validation Performance**: 0.6819 → 0.6889 (+1.02% gain)
3. **Expected Test Gain**: 0.72 → 0.73-0.74 (based on +5.6% val→test pattern)

### Performance Analysis
- **v1 → v2 Validation**: +1.02% improvement
- **Only change**: Dice loss weight adjustment
- **Training stability**: Converged at epoch 92 (vs 79 in v1)

### Submission Contents
```
submission_v2_f0.6889_codabench.zip
├── solution/
│   ├── ckpt.pth (20.2 MB) - epoch 92 checkpoint
│   ├── requirements.txt
│   ├── run.py (LMSA enabled)
│   └── microsegformer.py (with LMSA)
└── masks/ (100 PNG files - regenerated)
```

---

## Version Comparison

| Version | Val F-Score | Test F-Score | Key Config | Status |
|---------|-------------|--------------|------------|---------|
| v1 | 0.6819 | **0.72** | Dice=1.0 | ✅ Submitted |
| v2 | 0.6889 | TBD (0.73-0.74?) | Dice=1.5 | 📦 Ready |

**Next**: Upload v2 to Codabench to verify test performance!
