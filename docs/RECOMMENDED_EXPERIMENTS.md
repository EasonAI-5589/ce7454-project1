# 推荐实验配置 - 不会早停版本

**更新时间**: 2025-10-09
**目标**: 解决LR衰减和早停问题，充分训练模型

---

## 🎯 为什么要重新测试Dice 2.0/2.5？

### 原Dice 2.0的问题

**实验结果**: Val 0.6702 (68轮最佳，98轮早停)

**可能的问题**:
1. ⚠️ **早停太早**: patience=30，在68轮达到最佳后30轮就停了
2. ⚠️ **LR衰减过快**: CosineAnnealing，没有重启机会
3. ⚠️ **训练不充分**: 只训练了98轮，可能还有提升空间

**你的观点正确**:
- Dice 2.0可能不是不好，而是训练方式不对
- 配合Warm Restarts和充足训练时间，可能会更好

---

## 🚀 推荐实验配置（按优先级）

### ⭐⭐⭐⭐⭐ 优先级1: Dice 1.5 + Warm Restarts（最稳妥）

**文件**: `configs/lmsa_dice1.5_warm_restart.yaml`

**配置**:
```yaml
loss:
  dice_weight: 1.5  # 已验证最优

training:
  scheduler: CosineAnnealingWarmRestarts
  scheduler_params:
    T_0: 25
    T_mult: 2
  epochs: 200
  early_stopping_patience: 999  # 不早停
```

**理由**:
- Dice 1.5已验证是最优值(Val 0.6889)
- 但原实验在92轮后LR过低
- Warm Restarts + 不早停可能突破0.69

**预期**: Val 0.690-0.700

**运行**:
```bash
python main.py --config configs/lmsa_dice1.5_warm_restart.yaml
```

---

### ⭐⭐⭐⭐⭐ 优先级2: Dice 2.0 + Warm Restarts（重新验证）

**文件**: `configs/lmsa_dice2.0_warm_restart.yaml`

**配置**:
```yaml
loss:
  dice_weight: 2.0  # 重新测试

training:
  scheduler: CosineAnnealingWarmRestarts
  scheduler_params:
    T_0: 25
    T_mult: 2
  epochs: 200
  early_stopping_patience: 999  # 🔥 不早停
```

**理由**:
- 原Dice 2.0实验可能因早停和LR问题表现不佳
- Warm Restarts给多次收敛机会
- 200轮充分训练

**预期**:
- 如果你的假设正确: Val 0.690-0.705
- 如果确实不如1.5: Val 0.670-0.685

**运行**:
```bash
python main.py --config configs/lmsa_dice2.0_warm_restart.yaml
```

---

### ⭐⭐⭐⭐ 优先级3: Dice 2.5 + Warm Restarts（激进测试）

**文件**: `configs/lmsa_dice2.5_warm_restart_no_stop.yaml`

**配置**:
```yaml
loss:
  dice_weight: 2.5

model:
  dropout: 0.2  # 更强正则化

training:
  weight_decay: 2e-4
  scheduler: CosineAnnealingWarmRestarts
  scheduler_params:
    T_0: 30
    T_mult: 2
  warmup_epochs: 8  # 更长warmup
  epochs: 200
  early_stopping_patience: 999
```

**理由**:
- 如果Dice 2.0能改善，2.5也值得尝试
- 配合强正则化防止过拟合
- 更长warmup给复杂loss适应时间

**预期**:
- 乐观: Val 0.695-0.710
- 保守: Val 0.680-0.695

**运行**:
```bash
python main.py --config configs/lmsa_dice2.5_warm_restart_no_stop.yaml
```

---

### ⭐⭐⭐ 优先级4: Dice 1.5 + Higher LR（探索更高LR）

**文件**: `configs/lmsa_dice1.5_higher_lr.yaml`

**配置**:
```yaml
loss:
  dice_weight: 1.5

training:
  learning_rate: 1e-3  # +25% LR
  warmup_epochs: 10
  scheduler: CosineAnnealingLR
  scheduler_params:
    T_max: 150  # 慢衰减
  epochs: 200
  early_stopping_patience: 999
```

**理由**:
- 更高LR可能找到更好的优化路径
- 慢衰减避免后期LR过低

**预期**: Val 0.688-0.698

**运行**:
```bash
python main.py --config configs/lmsa_dice1.5_higher_lr.yaml
```

---

## 📊 实验对比表

| 配置 | Dice | LR策略 | Early Stop | 预期Val | 优先级 | 状态 |
|------|------|--------|------------|---------|--------|------|
| dice1.5_warm_restart | 1.5 | WarmRestarts | No (999) | 0.690-0.700 | ⭐⭐⭐⭐⭐ | ✅ Ready |
| dice2.0_warm_restart | 2.0 | WarmRestarts | No (999) | 0.690-0.705 | ⭐⭐⭐⭐⭐ | ✅ Ready |
| dice2.5_warm_restart_no_stop | 2.5 | WarmRestarts | No (999) | 0.695-0.710 | ⭐⭐⭐⭐ | ✅ Ready |
| dice1.5_higher_lr | 1.5 | SlowDecay | No (999) | 0.688-0.698 | ⭐⭐⭐ | ✅ Ready |

---

## 🎯 推荐运行顺序

### 方案A: 保守策略（最稳妥）

```bash
# Step 1: 先跑最稳的Dice 1.5 + Warm Restarts
python main.py --config configs/lmsa_dice1.5_warm_restart.yaml

# 如果结果>=0.695，可以停止
# 如果<0.695，继续Step 2

# Step 2: 尝试Dice 2.0 + Warm Restarts
python main.py --config configs/lmsa_dice2.0_warm_restart.yaml
```

---

### 方案B: 激进策略（追求最高分）

```bash
# 同时跑三个（如果有多GPU）
python main.py --config configs/lmsa_dice1.5_warm_restart.yaml &
python main.py --config configs/lmsa_dice2.0_warm_restart.yaml &
python main.py --config configs/lmsa_dice2.5_warm_restart_no_stop.yaml &
```

---

### 方案C: 你的建议（验证假设）

```bash
# 先验证Dice 2.0是否因早停而表现差
python main.py --config configs/lmsa_dice2.0_warm_restart.yaml

# 如果Dice 2.0能达到>=0.690，说明假设正确
# 然后尝试Dice 2.5
python main.py --config configs/lmsa_dice2.5_warm_restart_no_stop.yaml
```

---

## 🔥 关键改进点

### 1. Warm Restarts解决LR衰减

**原问题**:
```
原配置: LR从8e-4降到0.00064 (epoch 92)
问题: 后期LR太低，无法继续优化
```

**新方案**:
```
Warm Restarts: LR周期性重启
Epoch 0-25:   8e-4 → 1e-6
Epoch 26-75:  8e-4 → 1e-6 (重启!)
Epoch 76-175: 8e-4 → 1e-6 (重启!)
```

**效果**: 多次探索机会，避免局部最优

---

### 2. 不早停，充分训练

**原问题**:
```
Dice 2.0: 68轮最佳，98轮早停 (patience=30)
可能: 还未充分探索就停止了
```

**新方案**:
```yaml
early_stopping_patience: 999  # 实际上不会早停
epochs: 200  # 训练满200轮
```

**效果**: 给模型充分时间探索

---

### 3. 更强正则化（针对高Dice weight）

**配置**:
```yaml
# Dice 2.5配置
model:
  dropout: 0.2  # 从0.15提升

training:
  weight_decay: 2e-4  # 从1e-4提升
  warmup_epochs: 8  # 从5提升
```

**理由**: 高Dice weight可能更容易过拟合小目标

---

## 📈 预期结果

### 乐观情况（假设正确）

| 配置 | 预期Val | 预期Test |
|------|---------|----------|
| Dice 1.5 + WR | 0.695-0.700 | 0.735-0.745 |
| Dice 2.0 + WR | 0.695-0.705 | 0.735-0.750 |
| Dice 2.5 + WR | 0.700-0.710 | 0.740-0.755 |

### 保守情况（原结论成立）

| 配置 | 预期Val | 预期Test |
|------|---------|----------|
| Dice 1.5 + WR | 0.690-0.695 | 0.730-0.740 |
| Dice 2.0 + WR | 0.680-0.690 | 0.720-0.735 |
| Dice 2.5 + WR | 0.675-0.690 | 0.715-0.735 |

---

## ⚠️ 注意事项

### 1. 训练时间

**不早停 + 200轮**:
- 每轮约3-4分钟（GPU）
- 总时间: 10-13小时/实验
- 建议挂机过夜

### 2. 监控指标

**重点关注**:
- Val F-Score最高点
- 是否在多个重启周期都有提升
- Train-Val Gap是否合理

### 3. 何时停止

**提前停止条件**:
- 如果前100轮Val F-Score没有超过0.685，可能需要调整
- 如果Train-Val Gap > 0.05，过拟合严重

---

## 💡 实验假设验证

### 你的假设

> Dice 2.0不是效果不好，而是早停了，没有达到最优效果

**验证方法**:
1. 运行 `lmsa_dice2.0_warm_restart.yaml`
2. 对比结果:
   - 如果Val >= 0.690: **假设成立** ✅
   - 如果Val < 0.685: **假设不成立** ❌

### 如果假设成立

**意味着**:
- Dice weight可以继续探索2.0-2.5范围
- 早停和LR策略比Dice weight本身更重要
- Dice 2.5 + WR可能是最优解

### 如果假设不成立

**意味着**:
- Dice 1.5确实是最优值
- 应该专注于其他超参数（LR, 正则化）
- 不再增加Dice weight

---

**总结**: 所有配置已准备好，可以直接运行。推荐从 `lmsa_dice2.0_warm_restart.yaml` 开始验证你的假设！
