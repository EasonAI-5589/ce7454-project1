# CE7454 Face Parsing - 快速开始指南

## 🚀 服务器环境配置

### 方法1: 使用自动脚本 (推荐)
```bash
# 1. 上传代码到服务器
# 2. 运行环境配置脚本
bash setup_env.sh

# 3. 激活环境
conda activate ce7454
```

### 方法2: 手动配置
```bash
# 1. 创建环境
conda create -n ce7454 python=3.9 -y
conda activate ce7454

# 2. 安装PyTorch (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3. 安装其他依赖
pip install numpy pillow matplotlib tqdm

# 4. 验证
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## 📊 开始训练

### 默认配置训练
```bash
python main.py
```

### 自定义配置
```bash
python main.py --config configs/main.yaml --device cuda
```

### 从检查点恢复
```bash
python main.py --resume checkpoints/best_model.pth
```

## 🧪 测试推理

```bash
# 生成测试集预测
python -m src.inference \
    --model checkpoints/best_model.pth \
    --data data \
    --output predictions \
    --zip

# 这会生成:
# - predictions/*.png (预测掩码)
# - submission.zip (Codabench提交文件)
```

## 📂 数据结构要求

```
data/
├── train/
│   ├── images/  # 训练图像 (.jpg)
│   └── masks/   # 训练掩码 (.png)
└── test/
    └── images/  # 测试图像 (.jpg)
```

## ✅ 关键bug已修复

- ✅ NumPy负步长错误 (dataset.py:71-72) - 添加.copy()
- ✅ 验证集增强bug - 训练/验证正确分离
- ✅ MPS设备兼容性 - pin_memory仅CUDA启用
- ✅ 推理模块导入 - 更新到MicroSegFormer

## 🎯 预期训练时间 (单GPU)

- **V100**: ~2-3小时 (100 epochs)
- **A100**: ~1-2小时 (100 epochs)
- **RTX 3090**: ~3-4小时 (100 epochs)

## 📝 提交检查清单

- [ ] 训练完成，验证集F-Score > 0.75
- [ ] 生成测试集预测
- [ ] 创建submission.zip
- [ ] 上传到Codabench
- [ ] 检查排行榜分数
