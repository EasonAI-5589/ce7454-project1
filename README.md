# CE7454 Face Parsing - MicroSegFormer

Face parsing with 19-class segmentation using MicroSegFormer (1.72M parameters).

**Deadline**: October 14, 2024 | **Status**: ✅ Production Ready

## ✅ 最新修复 (Oct 5, 2025)

所有关键bug已修复，代码可直接训练：

1. ✅ NumPy负步长错误 - 修复训练崩溃
2. ✅ 验证集增强bug - 修复验证指标
3. ✅ 推理模块导入错误 - 修复测试预测
4. ✅ CUDA设备兼容性 - 服务器可用

## 🚀 服务器快速开始

```bash
# 1. Clone代码
git clone https://github.com/EasonAI-5589/ce7454-project1.git
cd ce7454-project1

# 2. 配置环境
bash setup_env.sh
conda activate ce7454

# 3. 开始训练
python main.py
```

## 📝 常用命令

### 训练
```bash
python main.py                                    # 开始训练
python main.py --device cuda:0                    # 指定GPU
python main.py --resume checkpoints/best_model.pth # 恢复训练
```

### 测试
```bash
# 验证集测试
python test.py --checkpoint checkpoints/best_model.pth

# 生成测试集预测 (Codabench提交)
python -m src.inference --model checkpoints/best_model.pth --data data --output predictions --zip
```

### 监控
```bash
tail -f checkpoints/microsegformer_*/training_log.txt  # 查看日志
nvidia-smi -l 1                                        # GPU监控
```

## 📊 配置说明

- **模型**: MicroSegFormer (1,721,939参数)
- **数据**: 1000训练图像 + 100验证图像
- **训练时间**: 4-6小时 (V100/A100)
- **目标F-Score**: > 0.80

## 📁 项目结构

```
ce7454-project1/
├── setup_env.sh          # 环境配置脚本 (新)
├── main.py               # 训练入口
├── test.py               # 评估脚本
├── src/
│   ├── dataset.py        # 数据加载 (已修复)
│   ├── inference.py      # 推理 (已修复)
│   ├── trainer.py        # 训练循环
│   └── models/microsegformer.py
├── configs/main.yaml     # 训练配置
└── data/                 # 数据集
```

## 🔧 故障排查

**CUDA内存不足**
```yaml
# configs/main.yaml
batch_size: 4  # 降低batch size
```

**检查环境**
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## 📚 文档

- **[QUICK_START.md](QUICK_START.md)** - 服务器详细配置
- **[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** - 完整训练指南
- **[docs/CODABENCH_README.md](docs/CODABENCH_README.md)** - 提交说明

---

**CE7454 Deep Learning for Data Science | NTU 2024**
