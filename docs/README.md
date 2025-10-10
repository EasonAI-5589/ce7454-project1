# Documentation Index

This directory contains all project documentation, analysis, and guides for the CE7454 Face Parsing Project.

**Project Status**: Val F-Score **0.7041** ⭐ | Test F-Score **TBD**

---

## 📂 Documentation Overview

### 🎯 Quick Start

| Document | Purpose | When to Use |
|----------|---------|-------------|
| [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) | Project introduction & goals | Start here |
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | How to train models | Training new models |
| [SUBMISSION_WORKFLOW.md](SUBMISSION_WORKFLOW.md) | ⭐ **Generate Codabench submissions** | Creating submission packages |

### 🏆 Breakthrough Analysis (Val 0.7041)

| Document | Content | Key Insights |
|----------|---------|--------------|
| [BREAKTHROUGH_0.7041_ANALYSIS.md](BREAKTHROUGH_0.7041_ANALYSIS.md) | ⭐ **Latest breakthrough** | LR=8e-4, warmup, reduced regularization |
| [BREAKTHROUGH_0.7041_ANALYSIS.png](BREAKTHROUGH_0.7041_ANALYSIS.png) | Training curves visualization | F-Score, Loss, LR, Train-Val Gap |

### 📊 Experimental Analysis

| Document | Focus | Findings |
|----------|-------|----------|
| [DICE_WEIGHT_ANALYSIS.md](DICE_WEIGHT_ANALYSIS.md) | Dice loss weight optimization | Optimal: 1.5 (not 2.0 or 2.5) |
| [LR_OVERFITTING_RELATIONSHIP.md](LR_OVERFITTING_RELATIONSHIP.md) | Learning rate vs overfitting | Low LR causes MORE overfitting |
| [RECOMMENDED_EXPERIMENTS.md](RECOMMENDED_EXPERIMENTS.md) | Next experiments to run | Warm Restarts, Dice 2.0 retesting |
| [WHY_LATE_TRAINING_STALLS.md](WHY_LATE_TRAINING_STALLS.md) | Training degradation analysis | LR decay causes overfitting |
| [COMPLETE_RESULTS_ANALYSIS.md](COMPLETE_RESULTS_ANALYSIS.md) | All experiment results summary | Comprehensive comparison |

### 🔧 Configuration & Optimization

| Document | Topic | Details |
|----------|-------|---------|
| [LR_OPTIMIZATION_CONFIGS.md](LR_OPTIMIZATION_CONFIGS.md) | Learning rate configs | CosineAnnealing, Warmup |
| [HYPERPARAMETER_OPTIMIZATION.md](HYPERPARAMETER_OPTIMIZATION.md) | Hyperparameter tuning guide | LR, batch size, weight decay |
| [GPU_OPTIMIZATION.md](GPU_OPTIMIZATION.md) | GPU utilization & speed | Kornia GPU augmentation |
| [CONTINUE_TRAINING.md](CONTINUE_TRAINING.md) | Resume from checkpoints | Continue training strategies |

### 🧪 Ablation Studies

| Document | Study | Conclusions |
|----------|-------|-------------|
| [ABLATION_STUDY.md](ABLATION_STUDY.md) | Component effectiveness | LMSA, Dice loss, augmentation |
| [ACCURACY_VS_FSCORE.md](ACCURACY_VS_FSCORE.md) | Metrics comparison | F-Score > Accuracy for imbalanced data |
| [TRAINING_RESULTS_SUMMARY.md](TRAINING_RESULTS_SUMMARY.md) | All training runs | Historical performance tracking |

### 🏗️ Architecture Documentation

| Document | Content | For |
|----------|---------|-----|
| [architecture_overview.md](architecture_overview.md) | MicroSegFormer architecture | Understanding model design |
| [vit_vs_segformer_comparison.md](vit_vs_segformer_comparison.md) | ViT vs SegFormer | Architecture comparison |
| [comparison_with_lightweight_models.md](comparison_with_lightweight_models.md) | Model comparisons | Benchmarking |

### 📝 Project Information

| Document | Content |
|----------|---------|
| [CE7454 Project 1 CelebAMask Face Parsing.md](CE7454%20Project%201%20CelebAMask%20Face%20Parsing.md) | Original project brief |
| [CODABENCH_README.md](CODABENCH_README.md) | Codabench submission guide |

---

## 🎯 Key Findings Summary

### 1. **Optimal Dice Weight: 1.5**
- Tested: 1.0, 1.5, 2.0, 2.5
- Winner: **1.5** (Val 0.6889 → 0.7041 with other improvements)
- Why: Balances CE loss (large objects) with Dice loss (small objects)

### 2. **Higher Learning Rate + Warmup**
- Previous: 4e-4 (no warmup)
- Breakthrough: **8e-4 with 5-epoch warmup**
- Result: +2.2% Val F-Score improvement
- Insight: Previous model was under-exploring solution space

### 3. **Reduced Regularization**
- Previous: dropout=0.2, weight_decay=2e-4 (OVER-regularized)
- Breakthrough: **dropout=0.15, weight_decay=1e-4**
- Result: Higher Train (0.7321) AND Val (0.7041) scores
- Insight: Small Train-Val gap NOT always good - may indicate under-capacity

### 4. **LR Decay Causes Overfitting**
- Observation: When LR < 6e-4, Train-Val gap increases dramatically
- Epoch 92: LR=6.4e-4, Gap=0.98%
- Epoch 112: LR=5.6e-4, Gap=9.29%
- Solution: Warm Restarts to periodically reset LR

### 5. **GPU Augmentation (Kornia)**
- Faster training (augmentation on GPU, not CPU bottleneck)
- Better efficiency without accuracy loss
- Used in breakthrough model (Val 0.7041)

---

## 📈 Performance Progression

| Checkpoint | Date | Val F-Score | Test F-Score | Key Features |
|------------|------|-------------|--------------|--------------|
| v1 (baseline) | Oct 8 | 0.6819 | **0.72** | Dice=1.0 |
| v2 (Dice opt) | Oct 9 | 0.6889 | TBD | Dice=1.5 |
| v3 (breakthrough) | Oct 9 | **0.7041** ⭐ | TBD | LR=8e-4, warmup, GPU aug |

**Validation → Test Pattern**: Typically +5-6% (v1: 0.6819 → 0.72 = +5.6%)

**Expected Test Score (v3)**: **0.74-0.75**

---

## 🚀 Next Steps

1. ✅ **Upload v3 to Codabench** - submission_v3_f0.7041_codabench.zip ready
2. **Try Warm Restarts** - Prevent late-stage overfitting
3. **Test Dice 2.0 with Warm Restarts** - Validate hypothesis
4. **Explore higher LR (1e-3)** - With longer warmup (10 epochs)

---

## 📁 File Statistics

Total documents: 22 files, 7,594 lines

**Categories**:
- Experimental Analysis: 9 docs
- Configuration Guides: 4 docs
- Architecture: 3 docs
- Quick Start: 3 docs
- Project Info: 2 docs
- Visualization: 1 image

---

## 🔗 Related Directories

- **Checkpoints**: `../checkpoints/` - All trained models
- **Configs**: `../configs/` - Training configurations
- **Submissions**: `../submissions/` - Codabench submission packages
- **Source**: `../src/` - Training and model code
