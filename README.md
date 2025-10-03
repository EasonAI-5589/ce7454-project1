# CE7454 Face Parsing - MicroSegFormer

Face parsing project using MicroSegFormer (1.72M parameters).

**Deadline**: October 14, 2024 11:59 PM
**Status**: Ready for Training ✅

## 🎯 Project Overview

- **Task**: Face parsing with 19-class semantic segmentation
- **Dataset**: CelebAMask-HQ mini (1000 train + 100 val images)
- **Model**: MicroSegFormer - Hierarchical Transformer
- **Parameters**: 1,723,027 (94.6% of 1,821,085 limit)
- **Target**: F-Score > 0.80

## 🚀 Quick Start (One Command)

```bash
# Start training immediately
./quick_start.sh train
```

That's it! Training will begin with optimal settings.

## 📁 Setup

### 1. Clone & Install
```bash
git clone https://github.com/EasonAI-5589/ce7454-project1.git
cd ce7454-project1
pip install -r requirements.txt
```

### 2. Dataset Setup

Download from [CodaBench](https://www.codabench.org/competitions/4381/):
```bash
# Extract dataset
unzip dev-public.zip

# Verify structure
data/
├── train/
│   ├── images/  # 1000 images
│   └── masks/   # 1000 masks
```

### 3. Train Model
```bash
# Start training
./quick_start.sh train

# Or with Python
python main.py --config configs/main.yaml
```

## 📊 Model Architecture

**MicroSegFormer** - Hierarchical Transformer for Face Parsing

```
Input (512×512×3)
    ↓
Hierarchical Encoder (4 stages)
├── Stage 1: 64 channels  → 128×128
├── Stage 2: 128 channels → 64×64
├── Stage 3: 256 channels → 32×32
└── Stage 4: 512 channels → 16×16
    ↓
Multi-Scale Fusion
├── Skip connections
└── Self-attention modules
    ↓
Lightweight MLP Decoder
    ↓
Output (512×512×19)
```

**Key Features**:
- 1,723,027 parameters (94.6% usage)
- Hierarchical transformer encoder
- Multi-scale feature fusion
- Efficient attention mechanism
- Optimized for face parsing

## 🎯 Training Configuration

**Default settings** (configs/main.yaml):
- **Epochs**: 150 with early stopping (patience=20)
- **Batch Size**: 8
- **Optimizer**: AdamW (lr=1e-3, wd=1e-4)
- **Scheduler**: CosineAnnealingLR with 5-epoch warmup
- **Loss**: CrossEntropy (1.0) + Dice (0.5)
- **Augmentation**: Flip, rotation, color jitter, scaling

**Expected Training Time**: 4-6 hours on V100/A100

## 📝 Commands Reference

### Training
```bash
# Start training
./quick_start.sh train

# Resume from checkpoint
./quick_start.sh resume checkpoints/best_model.pth

# Specify device
python main.py --device cuda:0
```

### Testing
```bash
# Test model on validation set
./quick_start.sh test checkpoints/best_model.pth

# Or with Python
python test.py --checkpoint checkpoints/best_model.pth
```

### Monitor Training
```bash
# Watch training logs
tail -f checkpoints/microsegformer_*/training_log.txt

# Check model parameters
python main.py --config configs/main.yaml  # Prints param count
```

## 📂 Project Structure

```
ce7454-project1/
├── main.py                  # Main training entry
├── test.py                  # Model evaluation
├── quick_start.sh           # One-command training
├── configs/
│   └── main.yaml           # Model configuration
├── src/
│   ├── trainer.py          # Training loop
│   ├── dataset.py          # Data loading
│   ├── augmentation.py     # Data augmentation
│   ├── inference.py        # Inference utilities
│   ├── utils.py            # Helper functions
│   └── models/
│       └── microsegformer.py  # Model architecture
├── docs/                   # Documentation
│   ├── PROJECT_OVERVIEW.md
│   ├── TRAINING_GUIDE.md
│   └── CODABENCH_README.md
└── data/                   # Dataset (not in git)
    └── train/
        ├── images/
        └── masks/
```

## 🎓 Training Outputs

After training, you'll get:

```
checkpoints/microsegformer_YYYYMMDD_HHMMSS/
├── best_model.pth          # Best F-Score checkpoint ⭐
├── last_model.pth          # Latest checkpoint
├── config.yaml             # Training config backup
└── training_log.txt        # Training metrics
```

**Checkpoint contains**:
- Model weights
- Optimizer state
- Scheduler state
- Best F-Score
- Training epoch

## 📊 Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| **F-Score** | > 0.80 | 🎯 Goal |
| **Parameters** | 1,723,027 | ✅ Under limit |
| **Training Time** | 4-6 hours | ✅ Optimized |
| **GPU Memory** | ~6GB | ✅ Efficient |

## 🔧 Troubleshooting

**CUDA Out of Memory**
```yaml
# In configs/main.yaml, reduce batch size
batch_size: 4  # or 2
```

**Training too slow**
```yaml
# Reduce workers if CPU bottleneck
num_workers: 2
```

**Low F-Score**
- Train longer (increase epochs to 200)
- Check data augmentation settings
- Verify data quality

## 📚 Documentation

Detailed guides in `docs/`:
- **[PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)** - Project structure, model info
- **[TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** - Complete training guide
- **[CODABENCH_README.md](docs/CODABENCH_README.md)** - Submission instructions

## 🎯 Next Steps

1. ✅ Setup environment
2. ✅ Download dataset
3. **Run training**: `./quick_start.sh train`
4. **Monitor progress**: Check training logs
5. **Test model**: Use best checkpoint
6. **Submit to CodaBench**

## 📋 Requirements

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **GPU**: 6GB+ VRAM recommended
- **Disk**: 2GB for dataset + checkpoints

See [requirements.txt](requirements.txt) for dependencies.

## 🏆 Competition Rules

- ✅ Single model (no ensemble)
- ✅ No pretrained weights
- ✅ No external data
- ✅ < 1,821,085 parameters

## 📖 References

- [CelebAMask-HQ Dataset](https://github.com/switchablenorms/CelebAMask-HQ)
- [Competition Page](https://www.codabench.org/competitions/4381/)
- [SegFormer Paper](https://arxiv.org/abs/2105.15203)

---

**CE7454 Deep Learning for Data Science | NTU 2024**
