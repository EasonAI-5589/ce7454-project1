# CE7454 Project 1 - CelebAMask Face Parsing

Face parsing project for CE7454 Deep Learning for Data Science course at NTU.

## 🎯 Project Overview
- **Task**: Face parsing with pixel-wise semantic labeling
- **Dataset**: CelebAMask-HQ mini dataset (1000 train + 100 test pairs)
- **Image Resolution**: 512x512
- **Classes**: 19 facial components and accessories
- **Parameter Limit**: < 1,821,085 trainable parameters

## 📁 Dataset Setup
**⚠️ The dataset is NOT included in this repository due to size and licensing constraints.**

To set up the dataset:
1. Download `dev-public.zip` from CodaBench Files page
2. Extract to project root:
   ```bash
   unzip dev-public.zip
   # This creates data/train/ and data/test/ folders automatically
   ```
3. Verify directory structure:
   ```
   data/
   ├── train/
   │   ├── images/    # 1000 training images (.jpg)
   │   └── masks/     # 1000 corresponding masks (.png)
   └── test/
       └── images/    # 100 test images (.jpg, no masks)
   ```

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/EasonAI-5589/ce7454-project1.git
cd ce7454-project1
pip install -r requirements.txt
```

### 2. Test Components
```bash
python test.py
```

### 3. Train Model
```bash
python src/train.py
```

### 4. Generate Test Predictions
```bash
python src/inference.py --model outputs/exp_XXXXXX/best_model.pth --zip
```

## 📂 Project Structure
```
ce7454-project1/
├── src/
│   ├── model.py         # Lightweight U-Net implementation
│   ├── dataset.py       # Simple data loading (no torchvision)
│   ├── utils.py         # Training utilities
│   ├── train.py         # Main training script
│   └── inference.py     # Test prediction generation
├── test.py             # Component testing
├── data/               # Dataset (not in git)
├── outputs/            # Training outputs (not in git)
└── requirements.txt    # Minimal dependencies
```

## 🏗️ Model Architecture
- **Type**: Lightweight U-Net
- **Parameters**: ~800K (well within 1.82M limit)
- **Loss**: Combined CrossEntropy + Dice Loss
- **Optimizer**: AdamW with Cosine Annealing
- **Augmentation**: Horizontal flip only (simple but effective)

## ✨ Key Features
- **Minimal Dependencies**: Only PyTorch, PIL, NumPy
- **No External Libraries**: No torchvision, sklearn, opencv
- **Simple but Effective**: Focused on working solution
- **Parameter Efficient**: Well under the 1.8M limit
- **Easy to Debug**: Clear, simple code structure

## 📊 Expected Results
- **Training Time**: ~2-4 hours on single GPU
- **Target F-Score**: 0.75+ (competitive performance)
- **Parameter Count**: ~800K (safe margin)

## 🎯 Codabench Submission
The inference script automatically creates the correct format:
```
submission.zip
├── solution/           # Empty folder (required)
└── masks/             # Single-channel PNG predictions
    ├── 0001.png
    ├── 0002.png
    └── ...
```

## 🔧 Troubleshooting

### Common Issues
1. **Import Errors**: Run `python test.py` to check all components
2. **CUDA Errors**: Model works on both GPU and CPU
3. **Data Loading**: Ensure data structure matches exactly

### Performance Tips
- Use GPU if available (significantly faster)
- Increase batch size if you have more GPU memory
- Training typically converges around epoch 60-80

## 📋 Requirements
- Python 3.8+
- PyTorch 2.0+
- PIL (Pillow)
- NumPy
- 4GB+ GPU memory (recommended)

## 🎓 Academic Notes
- No pretrained models allowed
- No model ensembles allowed
- No external data allowed
- Training from scratch only
- Must stay under parameter limit

## 📞 Support
If you encounter issues:
1. Run `python test.py` first
2. Check data structure
3. Verify requirements are installed

---
*Built for CE7454 Deep Learning for Data Science, NTU*