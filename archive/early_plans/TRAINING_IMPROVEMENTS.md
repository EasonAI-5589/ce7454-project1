# Training Improvements Summary

## Current Status (Oct 5, 2025)

- **Current Val F-Score**: 0.648 (Epoch 98)
- **Target**: 0.75+ (Ideal: 0.80+)
- **Gap**: ~10-15% improvement needed

## ✅ Completed Fixes

### 1. F-Score Calculation Bug (CRITICAL)
**Problem**: Calculated F-Score for all 19 classes, even if not present in GT
**Fix**: Now only computes for classes in `np.unique(target)` - matches Codabench exactly
**Impact**: Validation metrics now align with competition evaluation

### 2. F-Score Averaging Bug
**Problem**: Weighted mean F-Score by batch size, but F-Score is already averaged
**Fix**: Changed `f_scores.update(f_score, images.size(0))` → `f_scores.update(f_score, 1)`
**Impact**: Corrected training metrics

## ⚠️ Critical Issues Identified

### 1. Class Imbalance (16.3x ratio)
**Analysis**:
```
Class  0 (skin):       28.39% -> weight: 0.003
Class 13 (background): 31.53% -> weight: 0.002
Class 16 (rare class):  0.00% -> weight: 16.347  ⚠️
Class  3 (l_eye):       0.32% -> weight: 0.229
Class  4 (r_eye):       0.23% -> weight: 0.315
```

**Recommendation**: Add class weights to CrossEntropyLoss
```python
# In src/utils.py CombinedLoss.__init__
class_weights = torch.tensor([...])  # computed from training data
self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
```

### 2. Training Configuration

**Current settings** (configs/main.yaml):
- Epochs: 150
- LR: 1.5e-3
- Weight decay: 5e-4
- Scheduler: CosineAnnealingLR
- Early stopping: 30 epochs

**Issues**:
- No warmup scheduler implemented (config says warmup_epochs: 5, but not used)
- No label smoothing
- High weight decay (5e-4) may over-regularize

## 🎯 Recommended Improvements (Priority Order)

### Priority 1: Add Class Weights
```python
# Compute from training data
python scripts/compute_class_weights.py

# Apply in CombinedLoss
self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
```

**Expected improvement**: +3-5% F-Score

### Priority 2: Implement LR Warmup
```python
# In trainer.py
from torch.optim.lr_scheduler import LinearLR, SequentialLR

warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
main_scheduler = CosineAnnealingLR(optimizer, T_max=epochs-warmup_epochs)
scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], [warmup_epochs])
```

**Expected improvement**: +1-2% F-Score (better convergence)

### Priority 3: Add Label Smoothing
```python
self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
```

**Expected improvement**: +0.5-1% F-Score (better generalization)

### Priority 4: Optimize Training Settings
```yaml
training:
  epochs: 200  # Increase from 150
  learning_rate: 1e-3  # Reduce from 1.5e-3
  weight_decay: 1e-4   # Reduce from 5e-4 (less regularization)
  early_stopping_patience: 40  # Increase from 30
```

**Expected improvement**: +1-2% F-Score

## 📊 Expected Results

With all improvements:
- **Conservative estimate**: Val F-Score 0.70-0.73
- **Realistic target**: Val F-Score 0.75-0.78
- **Optimistic**: Val F-Score 0.80+

## 🚀 Quick Start Commands

```bash
# 1. Compute class weights
python -m src.dataset compute_weights

# 2. Update config with recommended settings
# Edit configs/main.yaml

# 3. Train with improvements
python main.py --config configs/main.yaml

# 4. Monitor training
tensorboard --logdir checkpoints/
```

## 📝 Notes

- Current model: MicroSegFormer (1.72M params, 94.6% of limit)
- Data augmentation is already strong (no changes needed)
- Training time: ~3-4 hours on A100 (150 epochs)
- Submission deadline: Oct 14, 2025
