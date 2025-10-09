# 如何继续训练模型

## 📋 概述

当训练过早停止或者想要在最佳checkpoint基础上继续优化时,可以使用继续训练功能。

## 🎯 使用场景

1. **Early stopping太早** - 模型还有潜力但被提前终止
2. **想要微调** - 在最佳checkpoint基础上用不同超参数继续训练
3. **过拟合后优化** - 增加正则化重新训练
4. **改变学习率策略** - 用更小的学习率精细调整

## 📝 两种继续训练方式

### 方式1: 保留优化器状态 (常规继续)

**适用场景**: 训练被意外中断,想要无缝继续

```bash
python continue_training.py \
    --checkpoint checkpoints/microsegformer_20251008_025917/best_model.pth \
    --config configs/lmsa_continue_from_best.yaml \
    --device cuda
```

**特点**:
- ✅ 保留Adam优化器的momentum
- ✅ 保留学习率调度器状态
- ✅ 从中断的epoch继续计数
- ⚠️  如果模型已经过拟合,可能效果不好

---

### 方式2: 重置优化器状态 (推荐)

**适用场景**: 训练过拟合了,想要escape local minimum

```bash
python continue_training.py \
    --checkpoint checkpoints/microsegformer_20251008_025917/best_model.pth \
    --config configs/lmsa_continue_from_best.yaml \
    --reset_optimizer \
    --device cuda
```

**特点**:
- ✅ 只保留模型权重
- ✅ 优化器状态重新初始化
- ✅ 学习率调度器重新开始
- ✅ Epoch从0重新计数
- ✅ 更容易escape过拟合的局部最优

---

## 🔧 配置文件说明

已经为你创建好了 `configs/lmsa_continue_from_best.yaml`:

```yaml
experiment:
  name: lmsa_continue_v2
  description: Continue from best model (0.6889) with enhanced regularization

model:
  dropout: 0.20  # ⬆️ 从 0.15 增加到 0.20,防止过拟合

training:
  epochs: 200  # 目标训练到 200 epochs
  learning_rate: 4e-4  # ⬇️ 降低学习率,从 8e-4 到 4e-4
  weight_decay: 2e-4  # ⬆️ 增加正则化,从 1e-4 到 2e-4
  early_stopping_patience: 30  # 设置 30 epochs patience

loss:
  dice_weight: 1.5  # 保持最佳配置
```

### 关键参数调整:

| 参数 | 原始值 | 新值 | 原因 |
|------|--------|------|------|
| dropout | 0.15 | **0.20** | 防止过拟合 |
| learning_rate | 8e-4 | **4e-4** | 更精细的优化 |
| weight_decay | 1e-4 | **2e-4** | 增强L2正则化 |
| early_stopping | null | **30** | 允许更多探索 |

---

## 🚀 完整操作流程

### Step 1: 确认最佳checkpoint

```bash
# 查看checkpoint信息
python -c "
import torch
ckpt = torch.load('checkpoints/microsegformer_20251008_025917/best_model.pth')
print(f'Epoch: {ckpt[\"epoch\"]}')
print(f'Best F-Score: {ckpt[\"best_f_score\"]:.6f}')
print(f'Val F-Score: {ckpt[\"val_f_score\"]:.6f}')
"
```

预期输出:
```
Epoch: 92
Best F-Score: 0.688861
Val F-Score: 0.688861
```

### Step 2: 在GPU服务器上运行

**推荐使用 `--reset_optimizer`** 因为原训练已经过拟合:

```bash
# SSH登录GPU服务器
ssh your_gpu_server

# 进入项目目录
cd ce7454-project1

# 拉取最新代码
git pull

# 运行继续训练 (推荐方式)
python continue_training.py \
    --checkpoint checkpoints/microsegformer_20251008_025917/best_model.pth \
    --config configs/lmsa_continue_from_best.yaml \
    --reset_optimizer \
    --device cuda \
    2>&1 | tee logs/continue_training_$(date +%Y%m%d_%H%M%S).log
```

### Step 3: 监控训练进度

```bash
# 实时查看日志
tail -f logs/continue_training_*.log

# 或者查看最新的checkpoint
ls -lt checkpoints/microsegformer_*/
```

### Step 4: 检查结果

训练完成后,检查新的结果:

```bash
python scripts/analyze_checkpoint.py \
    --checkpoint checkpoints/microsegformer_20251009_*/best_model.pth
```

---

## 📊 预期结果

基于分析,继续训练预期收益:

| 指标 | 当前 (Epoch 92) | 预期 (Epoch 200) | 增益 |
|------|----------------|------------------|------|
| Val F-Score | 0.6889 | **0.692-0.699** | +0.3%-1.0% |
| Test F-Score | TBD (0.73?) | **0.734-0.742** | +0.4%-1.2% |
| Train-Val Gap | 0.010 | **< 0.03** | 更健康 |

---

## ⚠️ 注意事项

### 1. 为什么推荐 `--reset_optimizer`?

从分析图可以看到:
- ✅ Epoch 92后验证分数持续下降
- ✅ Train-Val gap增大到6.5% (过拟合)
- ✅ 需要escape当前的局部最优

重置优化器可以:
- 清除Adam的momentum积累
- 让模型重新探索参数空间
- 结合更强正则化,防止回到过拟合状态

### 2. 如果继续训练效果不好怎么办?

可以随时停止并使用原始最佳checkpoint:
```bash
# 原始最佳模型始终可用
checkpoints/microsegformer_20251008_025917/best_model.pth
```

### 3. 多久检查一次结果?

建议:
- 前20 epochs: 每5 epochs检查一次
- 20-50 epochs: 每10 epochs检查一次  
- 50+ epochs: 根据early stopping自动决定

### 4. 如何调整超参数?

如果初步结果不理想,可以修改 `configs/lmsa_continue_from_best.yaml`:

**过拟合严重**:
```yaml
dropout: 0.25  # 进一步增加
weight_decay: 3e-4  # 进一步增加
```

**学习太慢**:
```yaml
learning_rate: 6e-4  # 适度提高
```

**还是下降**:
```yaml
early_stopping_patience: 50  # 给更多时间
```

---

## 🎓 进阶技巧

### 技巧1: 分阶段学习率

如果想要更精细的控制:

```bash
# Phase 1: 较高学习率探索 (50 epochs)
python continue_training.py \
    --checkpoint checkpoints/.../best_model.pth \
    --config configs/lmsa_continue_phase1.yaml \
    --reset_optimizer

# Phase 2: 降低学习率精调 (50 epochs)  
python continue_training.py \
    --checkpoint checkpoints/.../best_model.pth \
    --config configs/lmsa_continue_phase2.yaml \
    --reset_optimizer
```

### 技巧2: 尝试不同正则化

创建多个配置文件测试:
```bash
configs/lmsa_continue_dropout0.20.yaml
configs/lmsa_continue_dropout0.25.yaml
configs/lmsa_continue_weightdecay3e-4.yaml
```

并行训练,选择最佳:
```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python continue_training.py --config configs/lmsa_continue_dropout0.20.yaml &

# GPU 1
CUDA_VISIBLE_DEVICES=1 python continue_training.py --config configs/lmsa_continue_dropout0.25.yaml &
```

---

## ❓ 常见问题

**Q: 为什么不直接从epoch 92继续训练到300?**

A: 因为:
1. 原配置已经过拟合,继续会更差
2. 需要改变超参数(dropout, weight_decay)
3. 重置优化器可以escape局部最优

**Q: 如果新训练的结果比0.6889还差怎么办?**

A: 
1. 原始checkpoint永远保留,可以随时回退
2. 可以尝试不同的超参数组合
3. 可以使用原模型,0.6889已经很强了

**Q: 大概需要多久?**

A: 
- GPU训练: 约3-6小时 (100-150 epochs)
- 可以用early stopping自动停止
- 建议overnight运行

**Q: 如何知道是否真的有提升?**

A: 监控这些指标:
```
Best Val F-Score > 0.6889  ✅ 真的提升了
Train-Val Gap < 0.03  ✅ 没有过拟合
Recovery events > 5  ✅ 训练稳定
```

---

## 📚 相关文件

- 配置文件: [`configs/lmsa_continue_from_best.yaml`](../configs/lmsa_continue_from_best.yaml)
- 训练脚本: [`continue_training.py`](../continue_training.py)
- 分析脚本: [`scripts/analyze_checkpoint.py`](../scripts/analyze_checkpoint.py)
- 训练曲线: [`analysis_best_model_training.png`](../analysis_best_model_training.png)

---

## 🎯 总结

**推荐操作**:

```bash
# 最简单的命令 - 适合大多数情况
python continue_training.py \
    --checkpoint checkpoints/microsegformer_20251008_025917/best_model.pth \
    --config configs/lmsa_continue_from_best.yaml \
    --reset_optimizer \
    --device cuda
```

**预期**: Val 0.689 → 0.692-0.699 (+0.3%-1.0%)

**风险**: 低 (可回退)

**时间**: 3-6小时

祝训练顺利! 🚀
