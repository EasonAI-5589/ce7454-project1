# Face Parsing Models

This directory contains all model implementations for the face parsing task, organized for easy comparison and ablation studies.

## 📁 File Structure

```
src/models/
├── __init__.py              # Model registry and factory functions
├── simple_unet.py          # Baseline U-Net model
├── microsegformer.py       # Main Transformer model
├── ablation_models.py      # Ablation study variants
└── README.md               # This file
```

## 🎯 Available Models

### Baseline Models

| Model | File | Parameters | Usage | Description |
|-------|------|-----------|-------|-------------|
| `simple_unet` | simple_unet.py | 483K | 26.5% | Lightweight CNN U-Net baseline |
| `microsegformer` | microsegformer.py | 1.72M | 94.6% | Transformer-based main model |

### Ablation Study Models

| Model | Parameters | Usage | Purpose |
|-------|-----------|-------|---------|
| `microseg_no_attn` | 1.05M | 57.8% | Test attention importance (uses CNN instead) |
| `microseg_single` | 1.63M | 89.4% | Test multi-scale fusion (single-scale decoder) |
| `microseg_deep` | 1.82M | 100.0% | Test decoder complexity (deeper MLP) |

## 🚀 Usage

### Basic Usage

```python
from src.models import get_model, list_models, count_parameters

# List all available models
print(list_models())
# ['simple_unet', 'microsegformer', 'microseg_no_attn', 'microseg_single', 'microseg_deep']

# Create a model
model = get_model('microsegformer', num_classes=19)

# Count parameters
params = count_parameters(model)
print(f"Parameters: {params:,}")
```

### Direct Import

```python
from src.models import SimpleFaceParsingNet, MicroSegFormer

# Create specific models
simple_model = SimpleFaceParsingNet(n_classes=19)
transformer_model = MicroSegFormer(num_classes=19)
```

### Training with Different Models

```python
from src.models import get_model

# Train multiple models for comparison
for model_name in ['simple_unet', 'microsegformer', 'microseg_no_attn']:
    model = get_model(model_name, num_classes=19)
    # ... training code ...
```

## 🔬 Model Architecture Details

### SimpleFaceParsingNet (simple_unet)
- **Type**: CNN-based U-Net
- **Encoder**: 4 downsampling stages (16→32→64→128 channels)
- **Decoder**: 3 upsampling stages with skip connections
- **Advantage**: Fast inference, low memory
- **Disadvantage**: Limited capacity, local receptive field

### MicroSegFormer (microsegformer)
- **Type**: Transformer-based
- **Encoder**: 4-stage Mix Transformer
  - Overlapping patch embeddings
  - Efficient Self-Attention (reduced KV length)
  - Channel progression: 32→64→128→192
- **Decoder**: Lightweight MLP decoder
  - Multi-scale feature fusion
  - 4x upsampling to original resolution
- **Advantage**: Global receptive field, strong performance
- **Disadvantage**: More parameters

### Ablation Models

#### MicroSegFormer_NoAttention (microseg_no_attn)
- Replaces self-attention with depthwise separable convolution
- Tests: **Is attention mechanism important?**
- Expected: 5-10% performance drop

#### MicroSegFormer_SingleScale (microseg_single)
- Uses only last-stage features (no multi-scale fusion)
- Tests: **Are multi-scale features important?**
- Expected: 3-8% performance drop

#### MicroSegFormer_DeepMLP (microseg_deep)
- Uses deeper MLP decoder (3 layers vs 2 layers)
- Tests: **Is decoder complexity important?**
- Expected: Minimal difference (<2%)

## 📊 Expected Performance

Based on model capacity and architecture:

| Model | F-Score (Val) | Speed (FPS) | Best For |
|-------|--------------|-------------|----------|
| microsegformer | 0.80-0.85 | ~6.0 | **Best accuracy** |
| microseg_deep | 0.79-0.84 | ~5.8 | Testing decoder |
| microseg_single | 0.73-0.78 | ~6.2 | Ablation study |
| microseg_no_attn | 0.70-0.75 | ~6.5 | Ablation study |
| simple_unet | 0.68-0.73 | ~5.3 | **Fast baseline** |

## 🛠️ Testing Models

### Test All Models
```bash
python test_all_models.py
```

### Test Individual Model
```bash
python -c "from src.models.microsegformer import test_microsegformer; test_microsegformer()"
```

### Experiment Plan
```bash
python experiment_config.py
```

## 📈 Ablation Study Workflow

1. **Train baseline models**:
   ```bash
   python train.py --model simple_unet --epochs 150
   python train.py --model microsegformer --epochs 150
   ```

2. **Run ablation experiments**:
   ```bash
   python train.py --model microseg_no_attn --epochs 150
   python train.py --model microseg_single --epochs 150
   python train.py --model microseg_deep --epochs 150
   ```

3. **Compare results**:
   - Analyze validation F-Scores
   - Identify which components matter most
   - Guide future architecture decisions

## 🔑 Key Design Decisions

### Why MicroSegFormer?
1. **Transformer global context** - Better than CNN local receptive field
2. **Efficient Self-Attention** - Reduced computation via spatial reduction
3. **Multi-scale fusion** - Captures both details and semantics
4. **Parameter efficient** - Lightweight decoder (only 4% of total params)

### Why These Ablations?
1. **No Attention** - Tests if Transformer is overkill
2. **Single Scale** - Tests if simple upsampling works
3. **Deep Decoder** - Tests encoder vs decoder importance

## 📝 Adding New Models

To add a new model:

1. Create model file in `src/models/your_model.py`
2. Implement model class with `num_classes` parameter
3. Register in `__init__.py`:
   ```python
   from .your_model import YourModel
   MODEL_REGISTRY['your_model'] = YourModel
   ```
4. Test parameter count: `python test_all_models.py`

## ⚠️ Parameter Limit

**Maximum parameters: 1,821,085**

All models must stay within this limit. Use `count_parameters()` to verify.

## 📚 References

- SegFormer: [NeurIPS 2021](https://arxiv.org/abs/2105.15203)
- U-Net: [MICCAI 2015](https://arxiv.org/abs/1505.04597)
- CelebAMask-HQ: [Dataset](https://github.com/switchablenorms/CelebAMask-HQ)
