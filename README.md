# CE7454 Project 1 - CelebAMask Face Parsing

Face parsing project for CE7454 Deep Learning for Data Science course at NTU.

## Project Overview
- **Task**: Face parsing with pixel-wise semantic labeling
- **Dataset**: CelebAMask-HQ mini dataset (1000 train + 100 val pairs)
- **Image Resolution**: 512x512
- **Classes**: 19 facial components and accessories
- **Parameter Limit**: < 1,821,085 trainable parameters

## Dataset Setup
**⚠️ The dataset is NOT included in this repository due to size and licensing constraints.**

To set up the dataset:
1. Download `CelebAMask-HQ.zip` from the course platform
2. Extract to `data/` directory:
   ```bash
   unzip CelebAMask-HQ.zip
   mv CelebAMask-HQ/* data/
   ```
3. Verify directory structure:
   ```
   data/
   ├── train/
   │   ├── images/    # 1000 training images
   │   └── masks/     # 1000 corresponding masks
   └── val/
       ├── images/    # 100 validation images
       └── masks/     # 100 corresponding masks
   ```

## Quick Start
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Test implementation**:
   ```bash
   python quick_test.py
   ```

3. **Train model**:
   ```bash
   python src/train.py
   ```

4. **Generate predictions**:
   ```bash
   python src/inference.py --model_path outputs/checkpoints/best_model.pth
   ```

## Project Structure
```
ce7454-project1/
├── src/
│   ├── models/          # Model implementations
│   ├── config.py        # Configuration settings
│   ├── dataset.py       # Data loading
│   ├── train.py         # Training script
│   ├── inference.py     # Inference script
│   └── utils.py         # Utility functions
├── .claude/             # Claude workflow configuration
├── data/               # Dataset (not in git)
├── outputs/            # Model outputs (not in git)
└── experiments/        # Experiment logs (not in git)
```

## Model Architecture
- **Base**: Lightweight U-Net with ResNet-inspired backbone
- **Parameters**: ~1.2M (within 1.82M limit)
- **Loss**: Combined CrossEntropy + Dice + Focal Loss
- **Optimizer**: AdamW with Cosine Annealing

## Key Features
- Parameter-efficient model design
- Advanced data augmentation for small dataset
- Class imbalance handling
- Comprehensive evaluation metrics

## Results
- **Target**: F-Score > 0.8 on validation set
- **Codabench**: Aiming for top 30% ranking

## Important Notes
- No pretrained models allowed
- No model ensembles allowed
- No external data allowed
- Training from scratch only

## Submission Format
Codabench submission structure:
```
submission.zip
├── solution/
└── masks/
    ├── 0001.png
    ├── 0002.png
    └── ...
```

## License
This project is for academic purposes only. The CelebAMask-HQ dataset is subject to its original license terms.