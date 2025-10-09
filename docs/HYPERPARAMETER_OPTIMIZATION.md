# 超参数优化策略

**当前最佳**: Val F-Score 0.6889
**优化目标**: Val F-Score 0.70+

---

## 📊 当前最佳配置分析

### 已验证最优参数 ✅

```yaml
loss:
  ce_weight: 1.0
  dice_weight: 1.5  # ⭐ 关键参数，1.0→1.5带来+1.02%提升

training:
  learning_rate: 8e-4
  weight_decay: 1e-4
  batch_size: 32
  optimizer: AdamW
  scheduler: CosineAnnealingLR
  warmup_epochs: 5
  max_grad_norm: 1.0

model:
  dropout: 0.15

augmentation:
  horizontal_flip: 0.5
  rotation: 15  # 不要超过15度
  color_jitter: {brightness: 0.2, contrast: 0.2, saturation: 0.1}
  scale_range: [0.9, 1.1]
```

---

## 🔬 可优化的超参数空间

### 1. 损失函数权重 ⭐⭐⭐⭐⭐（最高优先级）

| 参数 | 当前值 | 建议测试范围 | 预期影响 | 优先级 |
|------|--------|-------------|----------|--------|
| **dice_weight** | 1.5 | **2.0, 2.5, 3.0** | +0.5-1.5% | ⭐⭐⭐⭐⭐ |
| ce_weight | 1.0 | 保持1.0 | - | - |
| use_focal | False | ❌ 已测试，-1.7% | 负面 | ❌ |

**实验设计**:
```yaml
# Experiment 1: Dice 2.0
dice_weight: 2.0  # +0.3-0.6%预期

# Experiment 2: Dice 2.5
dice_weight: 2.5  # +0.5-1.0%预期

# Experiment 3: Dice 3.0 (激进)
dice_weight: 3.0  # +0.2-0.8%预期，但可能过度

# Experiment 4: 动态权重
dice_weight: 1.5 → 2.5  # 训练过程中逐步增加
```

**理由**:
- Dice loss直接优化F-Score（评估指标）
- 小目标（眼睛、嘴巴）受益最大
- 历史数据证明单参数最大收益

---

### 2. 学习率策略 ⭐⭐⭐⭐

| 参数 | 当前值 | 建议测试 | 预期影响 | 优先级 |
|------|--------|---------|----------|--------|
| **learning_rate** | 8e-4 | **6e-4, 1e-3** | +0.2-0.5% | ⭐⭐⭐⭐ |
| warmup_epochs | 5 | **8, 10** | +0.1-0.3% | ⭐⭐⭐ |
| scheduler | CosineAnnealing | **CosineWarmRestarts** | +0.1-0.4% | ⭐⭐⭐ |

#### 实验A: 学习率扫描

```yaml
# Experiment 1: 略低LR（更稳定）
learning_rate: 6e-4
warmup_epochs: 8

# Experiment 2: 略高LR（更快收敛）
learning_rate: 1e-3
warmup_epochs: 10

# Experiment 3: 更长warmup
learning_rate: 8e-4
warmup_epochs: 10
```

**理由**:
- 当前8e-4可能不是最优
- 更长warmup可能帮助稳定训练
- Face parsing是精细任务，可能受益于更低LR

#### 实验B: Cosine Warm Restarts

```yaml
scheduler: CosineAnnealingWarmRestarts
scheduler_params:
  T_0: 30      # 每30轮重启
  T_mult: 2    # 重启周期翻倍
  eta_min: 1e-6
```

**优势**:
- 周期性重启可能跳出局部最优
- 多次收敛机会
- 对长训练(200 epochs)有帮助

**预期**: +0.1-0.4%

---

### 3. 正则化参数 ⭐⭐⭐

| 参数 | 当前值 | 建议测试 | 预期影响 | 优先级 |
|------|--------|---------|----------|--------|
| **dropout** | 0.15 | **0.1, 0.2, 0.25** | +0.1-0.3% | ⭐⭐⭐ |
| **weight_decay** | 1e-4 | **5e-5, 2e-4, 3e-4** | +0.1-0.3% | ⭐⭐⭐ |
| max_grad_norm | 1.0 | 0.5, 2.0 | +0.0-0.2% | ⭐⭐ |

#### 实验设计

**组合1: 轻正则化**（更高容量）
```yaml
dropout: 0.1
weight_decay: 5e-5
```
- 适合：数据量小(1000张)，模型需要记忆更多细节
- 风险：轻微过拟合

**组合2: 强正则化**（防过拟合）
```yaml
dropout: 0.25
weight_decay: 3e-4
```
- 适合：防止在小数据集上过拟合
- 风险：欠拟合

**组合3: 平衡**（当前+调整）
```yaml
dropout: 0.2
weight_decay: 2e-4
```
- 平衡方案，略增强正则化

**预期**: +0.1-0.3%

---

### 4. Batch Size与优化器参数 ⭐⭐⭐

| 参数 | 当前值 | 建议测试 | 预期影响 | 优先级 |
|------|--------|---------|----------|--------|
| **batch_size** | 32 | **24, 40, 48** | +0.1-0.4% | ⭐⭐⭐ |
| optimizer | AdamW | **AdamW + lookahead** | +0.1-0.3% | ⭐⭐ |

#### Batch Size实验

**小Batch (24)**:
```yaml
batch_size: 24
learning_rate: 6e-4  # 对应调低
```
- 优点：更多梯度更新，更强正则化效果
- 缺点：训练时间+25%

**大Batch (40-48)**:
```yaml
batch_size: 40
learning_rate: 1e-3  # 对应调高
```
- 优点：训练更稳定，速度更快
- 缺点：泛化性可能略差

**预期**: +0.1-0.4%

#### AdamW参数调整

```yaml
optimizer: AdamW
optimizer_params:
  betas: [0.9, 0.999]  # 当前
  # 测试更高momentum
  betas: [0.95, 0.999]  # 更平滑的优化路径
```

**预期**: +0.0-0.2%

---

### 5. 数据增强微调 ⭐⭐

| 参数 | 当前值 | 建议测试 | 预期影响 | 优先级 |
|------|--------|---------|----------|--------|
| rotation | 15 | **10, 12** | +0.0-0.2% | ⭐⭐ |
| scale_range | [0.9,1.1] | **[0.85,1.15], [0.95,1.05]** | +0.0-0.2% | ⭐⭐ |
| horizontal_flip | 0.5 | **0.6, 0.7** | +0.0-0.1% | ⭐ |

**注意**:
- ⚠️ 已验证过度增强会降低性能(-0.5%)
- 仅建议微小调整
- **不要超过**: rotation 15°, color_jitter 0.2

#### 保守调整方案

```yaml
augmentation:
  horizontal_flip: 0.6  # 稍微增加
  rotation: 12          # 稍微减少（更保守）
  scale_range: [0.92, 1.08]  # 更保守的范围
  color_jitter:
    brightness: 0.15    # 稍微减少
    contrast: 0.15      # 稍微减少
    saturation: 0.1     # 保持
```

**预期**: +0.0-0.2%

---

### 6. 高级技巧 ⭐⭐⭐

#### A. Label Smoothing

```yaml
loss:
  label_smoothing: 0.1  # 平滑one-hot标签
```

**原理**:
- 将hard label [0,1] 变为 [0.05, 0.95]
- 防止过度自信，提高泛化

**预期**: +0.1-0.3%

#### B. Mixup / CutMix (谨慎使用)

```yaml
augmentation:
  mixup_alpha: 0.2  # 混合两张图像
  # 或
  cutmix_alpha: 1.0  # 裁剪粘贴
```

**注意**: Face parsing可能不适合Mixup（破坏面部结构）

**预期**: -0.2 to +0.2% (不确定)

#### C. Stochastic Depth (针对Decoder)

```yaml
model:
  stochastic_depth: 0.1  # 随机丢弃层
```

**优点**: 类似dropout的正则化
**预期**: +0.1-0.2%

---

## 🎯 推荐实验计划

### Tier 1: 高优先级（必做）⭐⭐⭐⭐⭐

| 实验ID | 配置变化 | 预期提升 | GPU时间 |
|--------|---------|---------|---------|
| **E1** | Dice=2.0 | +0.3-0.6% | 6h |
| **E2** | Dice=2.5 | +0.5-1.0% | 6h |
| **E3** | Dice=2.5 + dropout=0.2 + wd=2e-4 | +0.6-1.2% | 6h |
| **E4** | Dice=3.0 | +0.2-0.8% | 6h |

**总时间**: 24小时
**预期最佳收益**: +0.6-1.2%

---

### Tier 2: 中优先级（建议）⭐⭐⭐⭐

| 实验ID | 配置变化 | 预期提升 | GPU时间 |
|--------|---------|---------|---------|
| **E5** | LR=6e-4 + warmup=8 | +0.2-0.5% | 6h |
| **E6** | LR=1e-3 + warmup=10 | +0.2-0.5% | 6h |
| **E7** | CosineWarmRestarts | +0.1-0.4% | 6h |
| **E8** | Label Smoothing=0.1 | +0.1-0.3% | 6h |

**总时间**: 24小时
**预期最佳收益**: +0.2-0.5%

---

### Tier 3: 低优先级（可选）⭐⭐

| 实验ID | 配置变化 | 预期提升 | GPU时间 |
|--------|---------|---------|---------|
| E9 | Batch=24 + LR=6e-4 | +0.1-0.3% | 8h |
| E10 | Batch=40 + LR=1e-3 | +0.1-0.3% | 5h |
| E11 | Dropout sweep (0.1,0.2,0.25) | +0.1-0.3% | 18h |

**总时间**: 31小时
**预期最佳收益**: +0.1-0.3%

---

## 📋 最优执行策略

### 策略A: 快速迭代（推荐）

**时间**: 3天
**目标**: Val 0.70+

```bash
# Day 1: Dice weight扫描
python main.py --config configs/lmsa_dice2.0_fresh.yaml
python main.py --config configs/lmsa_dice2.5_aggressive.yaml

# Day 2: 最佳Dice + LR优化
# 假设Dice 2.5最好，创建新config
python main.py --config configs/lmsa_dice2.5_lr6e-4.yaml

# Day 3: 最佳组合 + Label Smoothing
python main.py --config configs/lmsa_dice2.5_final.yaml
```

**预期结果**: Val 0.695-0.705

---

### 策略B: 网格搜索（全面）

**时间**: 5-7天
**目标**: 找到全局最优

```python
# 网格搜索配置
grid = {
    'dice_weight': [2.0, 2.5, 3.0],
    'learning_rate': [6e-4, 8e-4, 1e-3],
    'dropout': [0.1, 0.15, 0.2],
    'weight_decay': [1e-4, 2e-4]
}

# 总组合: 3 × 3 × 3 × 2 = 54个
# 智能剪枝后: ~15-20个关键组合
```

**不推荐**: 时间成本太高

---

## 🔧 快速配置生成器

### 配置1: Dice 2.5 + 强正则化（推荐）

```yaml
# configs/lmsa_dice2.5_strong_reg.yaml
loss:
  dice_weight: 2.5

model:
  dropout: 0.2

training:
  learning_rate: 6e-4
  weight_decay: 2e-4
  warmup_epochs: 8
```

**预期**: Val 0.695-0.705

---

### 配置2: Dice 2.0 + 优化LR

```yaml
# configs/lmsa_dice2.0_optim_lr.yaml
loss:
  dice_weight: 2.0

training:
  learning_rate: 1e-3
  warmup_epochs: 10
  scheduler: CosineAnnealingWarmRestarts
```

**预期**: Val 0.690-0.700

---

### 配置3: Dice 2.5 + Label Smoothing

```yaml
# configs/lmsa_dice2.5_label_smooth.yaml
loss:
  dice_weight: 2.5
  label_smoothing: 0.1

model:
  dropout: 0.2

training:
  learning_rate: 6e-4
  weight_decay: 2e-4
```

**预期**: Val 0.697-0.707

---

## 📊 预期提升路径

```
当前最佳: 0.6889
  ↓ +Dice 2.5
0.694-0.699
  ↓ +LR优化
0.696-0.701
  ↓ +Label Smoothing
0.698-0.703
  ↓ +正则化调整
0.700-0.705 ← 目标

Test分数预期: 0.74-0.75
```

---

## ⚠️ 注意事项

### 不要做的事情

1. ❌ **过度增强数据**: rotation > 15°, color_jitter > 0.2
2. ❌ **使用Focal Loss**: 已验证-1.7%
3. ❌ **改变Batch Size到极端值**: <16 或 >64
4. ❌ **同时改变太多参数**: 难以归因

### 实验最佳实践

1. ✅ **单变量对照**: 每次只改1-2个参数
2. ✅ **记录所有结果**: 包括失败的实验
3. ✅ **多次验证**: 重要配置跑2-3次确认
4. ✅ **Early stopping patience=100**: 充分训练

---

## 💡 最终建议

### 立即执行（已准备）

```bash
# Tier 1实验已配置完成
python main.py --config configs/lmsa_dice2.5_aggressive.yaml  # E3
python main.py --config configs/lmsa_dice2.0_fresh.yaml       # E1
```

### 下一步准备

如果Dice 2.5达到0.695-0.700，创建配置：
```yaml
# configs/lmsa_dice2.5_final.yaml
loss:
  dice_weight: 2.5
  label_smoothing: 0.1  # 新增

model:
  dropout: 0.2

training:
  learning_rate: 6e-4
  weight_decay: 2e-4
  warmup_epochs: 8
  scheduler: CosineAnnealingWarmRestarts
  scheduler_params:
    T_0: 30
```

**预期**: Val 0.70-0.71 → Test 0.74-0.75

---

**总结**: 超参数优化空间巨大，优先完成Dice weight实验后，再逐步优化LR、正则化、Label Smoothing等。

**最后更新**: 2025-10-09
