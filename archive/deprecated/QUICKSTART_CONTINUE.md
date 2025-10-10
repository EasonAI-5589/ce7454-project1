# 🚀 继续训练 - 快速开始

## 最简单的命令 (推荐)

```bash
python continue_training.py \
    --checkpoint checkpoints/microsegformer_20251008_025917/best_model.pth \
    --config configs/lmsa_continue_from_best.yaml \
    --reset_optimizer \
    --device cuda
```

## 三个文件已经准备好:

1. ✅ **配置文件**: `configs/lmsa_continue_from_best.yaml`
   - Dropout: 0.15 → **0.20** (防过拟合)
   - Learning rate: 8e-4 → **4e-4** (更精细)
   - Weight decay: 1e-4 → **2e-4** (更强正则化)

2. ✅ **训练脚本**: `continue_training.py`
   - 自动从checkpoint加载
   - 支持重置优化器
   - 保存完整训练历史

3. ✅ **详细文档**: `docs/CONTINUE_TRAINING.md`
   - 完整的使用说明
   - 常见问题解答
   - 进阶技巧

## 为什么要继续训练?

从训练曲线分析:
- 📉 当前训练在epoch 92达到最佳 (Val 0.6889)
- 📉 之后20个epoch持续下降 (过拟合)
- ✅ 但Learning Rate还有70%优化空间
- ✅ 有15次recovery事件,说明模型有潜力

## 预期收益

| 指标 | 当前 | 预期 | 提升 |
|------|------|------|------|
| Val F-Score | 0.6889 | 0.692-0.699 | +0.3%-1.0% |
| Test F-Score | ~0.73 | 0.734-0.742 | +0.4%-1.2% |

## 在GPU服务器上运行

```bash
# 1. SSH登录
ssh your_gpu_server

# 2. 进入项目
cd ce7454-project1

# 3. 拉取最新代码
git pull

# 4. 运行训练 (保存日志)
python continue_training.py \
    --checkpoint checkpoints/microsegformer_20251008_025917/best_model.pth \
    --config configs/lmsa_continue_from_best.yaml \
    --reset_optimizer \
    --device cuda \
    2>&1 | tee logs/continue_$(date +%Y%m%d_%H%M%S).log
```

## 监控进度

```bash
# 实时查看日志
tail -f logs/continue_*.log

# 查看最新checkpoint
ls -lt checkpoints/microsegformer_*/best_model.pth | head -3
```

## 注意事项

⚠️ **推荐使用 `--reset_optimizer`** 因为:
- 原训练已经过拟合 (Train-Val gap 6.5%)
- 重置可以escape局部最优
- 结合新的正则化参数效果更好

✅ **原始模型永远保留**:
- `checkpoints/microsegformer_20251008_025917/best_model.pth`
- 可以随时回退

⏱️ **预计时间**: 3-6小时 (100-150 epochs)

📊 **查看详细说明**: `docs/CONTINUE_TRAINING.md`

祝训练顺利! 🎯
