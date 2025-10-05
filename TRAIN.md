# 训练指南

## 快速开始

### 默认训练（推荐）
```bash
python main.py
```

### 指定配置文件
```bash
python main.py --config configs/main.yaml
```

### 指定GPU
```bash
python main.py --device cuda:0
```

### 从检查点恢复
```bash
python main.py --resume checkpoints/microsegformer_*/best_model.pth
```

## 监控训练

### 实时查看日志
```bash
tail -f checkpoints/microsegformer_*/training_log.txt
```

### 监控GPU
```bash
watch -n 1 nvidia-smi
```

## 当前配置 (A100优化)

- **Batch Size**: 64
- **Workers**: 16
- **Learning Rate**: 2e-3
- **Epochs**: 150
- **Early Stopping**: 20 epochs

预期GPU利用率: 85%+
