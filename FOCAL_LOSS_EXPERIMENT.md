# Focal Loss 实验计划

## 📚 Focal Loss 原理详解

### 什么是类别不平衡问题？

在CelebAMask-HQ人脸分割数据集中，像素分布极不均衡：

| 类别 | 像素占比 | 相对大小 |
|------|---------|---------|
| 背景 (Background) | ~60-70% | 1x (基准) |
| 皮肤 (Skin) | ~15-20% | 0.3x |
| 头发 (Hair) | ~10-15% | 0.2x |
| 嘴巴 (Mouth) | ~1-2% | 0.02x |
| 眼睛 (Eyes) | ~0.2% | 0.003x (300倍差距!) |
| 耳朵 (Ears) | ~0.4% | 0.006x |

**问题**: 传统Cross-Entropy Loss被大量背景像素主导，模型倾向于"偷懒"：
- 正确预测背景 → 损失下降明显
- 错误预测小目标 → 损失几乎不变
- 结果：模型忽略小目标！

### Focal Loss 如何解决？

**核心思想**: 动态调整每个样本的损失权重

```
标准CE Loss: L = -log(pt)
Focal Loss:  L = -α(1-pt)^γ * log(pt)

其中:
  pt = 模型预测该类别的概率
  α = 类别平衡因子 (typical: 0.25)
  γ = 聚焦参数 (typical: 2.0)
```

**工作机制**:

1. **简单样本** (模型预测准确, pt ≈ 1):
   ```
   pt = 0.95 (模型很确定)
   → (1-pt)^2 = 0.05^2 = 0.0025
   → 损失被降低到原来的 0.25%
   → 模型不再浪费时间在简单背景上！
   ```

2. **困难样本** (模型预测不准, pt ≈ 0.5):
   ```
   pt = 0.5 (模型不确定)
   → (1-pt)^2 = 0.5^2 = 0.25
   → 损失保持较高权重
   → 模型被迫关注这些难例！
   ```

3. **错误样本** (模型预测错误, pt ≈ 0):
   ```
   pt = 0.1 (模型预测错)
   → (1-pt)^2 = 0.9^2 = 0.81
   → 损失接近原值
   → 模型必须学会正确分类！
   ```

### 参数调优指南

#### γ (Gamma) - 聚焦强度

| γ值 | 效果 | 适用场景 |
|-----|------|---------|
| 0 | 退化为标准CE | 类别平衡数据 |
| 1 | 轻度聚焦 | 轻微不平衡 (10:1) |
| 2 | 标准聚焦 | 中度不平衡 (100:1) ⭐ 推荐 |
| 3 | 强力聚焦 | 严重不平衡 (1000:1) |
| 5 | 极端聚焦 | 极端不平衡 (>10000:1) |

**我们的数据**: 背景:眼睛 ≈ 300:1 → **γ=2.0 合适**

#### α (Alpha) - 类别平衡

| α值 | 含义 | 使用场景 |
|-----|------|---------|
| 0.25 | 正样本权重25% | 负样本多(1:3) ⭐ 标准值 |
| 0.5 | 正负样本等权重 | 样本大致平衡 |
| 0.75 | 正样本权重75% | 正样本多(3:1) |

**我们的数据**: 小目标是"正样本"，背景是"负样本" → **α=0.25-0.5**

## 🎯 实验设计

### 实验1: LMSA + Focal Loss (保守策略)

**配置文件**: `configs/lmsa_focal.yaml`

**关键参数**:
```yaml
loss:
  use_focal: true
  focal_alpha: 0.25    # 标准值
  focal_gamma: 2.0     # 标准值
  dice_weight: 1.0

training:
  learning_rate: 5e-4  # 略低于baseline
  epochs: 250
  warmup_epochs: 10
```

**预期效果**:
- Val F-Score: 0.71-0.73 (+4-7%)
- 小目标类别显著改善
- 训练稳定

**适合场景**:
- ✅ 首选方案
- ✅ 稳定性优先
- ✅ 时间充足 (4-5小时)

---

### 实验2: LMSA + Aggressive (激进策略)

**配置文件**: `configs/lmsa_aggressive.yaml`

**关键参数**:
```yaml
loss:
  use_focal: true
  focal_alpha: 0.5     # 更高权重
  focal_gamma: 2.5     # 更强聚焦
  dice_weight: 1.5     # Dice对小目标更好
  use_class_weights: true  # 双重平衡

training:
  learning_rate: 3e-4  # 更低LR
  batch_size: 24       # 更小batch
  epochs: 300
  dropout: 0.2         # 更强正则化
```

**预期效果**:
- Val F-Score: 0.73-0.76 (+7-12%)
- 可能达到0.75目标
- 训练可能不稳定

**风险**:
- ⚠️ 可能过拟合
- ⚠️ 可能损失震荡
- ⚠️ 训练时间更长 (5-6小时)

**适合场景**:
- 🎯 追求最高分数
- 🎯 有监控调整能力
- 🎯 deadline压力大

---

## 🚀 执行计划

### 方案A: 保守并行 (推荐)

```bash
# 同时运行两个实验
python main.py --config configs/lmsa_focal.yaml &
python main.py --config configs/lmsa_aggressive.yaml &
```

**优点**:
- 快速获得两组结果
- 对比分析最佳策略
- 降低风险

**缺点**:
- GPU资源需求大
- 需要监控两个实验

---

### 方案B: 串行稳健

```bash
# Step 1: 先跑保守版本
python main.py --config configs/lmsa_focal.yaml

# Step 2: 评估结果
# - 如果F-Score > 0.73 → 成功，继续优化
# - 如果F-Score < 0.71 → 参数调整

# Step 3: 根据结果决定是否跑激进版本
python main.py --config configs/lmsa_aggressive.yaml
```

**优点**:
- GPU资源节省
- 基于结果动态决策
- 风险可控

**缺点**:
- 总时间更长 (8-10小时)

---

## 📊 监控指标

### 训练前期 (Epoch 1-20)

**健康信号**:
- ✅ Train loss 快速下降
- ✅ Val loss 跟随下降
- ✅ Val F-Score > 0.3 by epoch 10
- ✅ 无NaN或Inf

**警告信号**:
- ⚠️ Val loss 不降反升
- ⚠️ Train/Val loss 差距>2.0
- ⚠️ Val F-Score < 0.2 by epoch 10

**行动**: 如有警告信号 → 降低γ到2.0或降低LR

---

### 训练中期 (Epoch 50-150)

**健康信号**:
- ✅ Val F-Score 稳定增长
- ✅ Val F-Score > 0.65 by epoch 100
- ✅ Train/Val F-Score差距 < 0.1

**警告信号**:
- ⚠️ Val F-Score 震荡 (±0.05)
- ⚠️ Train F-Score >> Val F-Score (+0.15)
- ⚠️ Val F-Score停滞不前

**行动**:
- 震荡 → 降低LR或减小γ
- 过拟合 → 增加dropout或早停
- 停滞 → 增加LR或增强augmentation

---

### 训练后期 (Epoch 150+)

**目标检查**:
- 🎯 Val F-Score > 0.73 → **成功！**
- ⚠️ Val F-Score 0.70-0.73 → 可接受
- ❌ Val F-Score < 0.70 → 需要重新设计

---

## 🔧 应急调参方案

### 如果训练不稳定 (Loss震荡)

```yaml
# 降低Focal Loss强度
focal_gamma: 2.0 → 1.5
focal_alpha: 0.25 → 0.2

# 降低学习率
learning_rate: 5e-4 → 3e-4

# 增加warmup
warmup_epochs: 10 → 20
```

---

### 如果过拟合 (Train>>Val)

```yaml
# 增强正则化
dropout: 0.15 → 0.25
weight_decay: 1e-4 → 2e-4

# 增强数据增强
rotation: 20 → 30
scale_range: [0.85, 1.15] → [0.8, 1.2]

# 减少模型容量
# (已经接近参数上限，此项慎用)
```

---

### 如果欠拟合 (Val<0.65持续50轮)

```yaml
# 增加模型学习能力
learning_rate: 5e-4 → 8e-4
dropout: 0.15 → 0.1

# 增加训练时间
epochs: 250 → 350

# 考虑模型架构调整
# (此时可能需要更大改动)
```

---

## ⏰ 时间规划

**当前**: 10月7日晚
**Deadline**: 10月14日

### 今晚 (10/7): 启动训练
- ✅ 创建配置文件
- ✅ 启动实验1 (focal)
- ⏳ 监控前20个epoch
- ⏳ 如稳定，启动实验2 (aggressive)

### 明天 (10/8): 结果分析
- 查看150-200 epoch结果
- 选择最佳配置
- 如需要，启动微调实验

### 10/9-10/11: 最终优化
- 测试集推理
- 模型集成(如时间允许)
- Codabench提交

### 10/12-10/14: 报告撰写
- 技术报告
- 代码整理
- 最终提交

---

## 📝 期望贡献点 (报告用)

### 技术创新
1. **架构层面**: LMSA模块 (轻量级多尺度注意力)
2. **训练层面**: Focal Loss应对类别不平衡
3. **组合效果**: 架构+训练策略的协同优化

### 实验分析
1. **消融实验**: Baseline → +LMSA → +Focal Loss
2. **参数研究**: γ和α对小目标的影响
3. **类别分析**: 各类别F-Score的改善程度

### 学术价值
- 系统性解决方案 (不是调参碰运气)
- 理论支撑 (Focal Loss原理)
- 可复现性 (详细配置和分析)

---

## 🎓 总结

**Focal Loss核心价值**:
让模型"聪明"地分配学习资源 - 多关注小目标和困难样本，少浪费时间在简单背景上。

**我们的策略**:
LMSA (架构) + Focal Loss (训练) = 针对小目标的双重优化

**预期效果**:
0.6819 → 0.73-0.76 (提升7-12%)

**立即行动**:
```bash
python main.py --config configs/lmsa_focal.yaml
```

Good luck! 🚀
