# 模型优化策略 - 目标提升F-Score至0.75+

## 当前状态分析

### 性能指标
- **当前最佳 Val F-Score**: 0.648 (Epoch 98)
- **目标**: 0.75+ (需要提升 **+16%**)
- **过拟合程度**: 11.3% (Train 0.704 vs Val 0.632)

### 发现的问题

#### 🚨 问题1: 严重过拟合 (11.3% gap)
**原因**:
- 模型完全没有dropout
- Weight decay可能不足 (当前5e-4)
- 数据增强虽强但模型容量太大

**已实施的解决方案**:
- ✅ 添加0.15 progressive dropout到所有层
- ✅ 降低weight decay到1e-4
- ✅ 降低学习率 1.5e-3 → 8e-4

#### 🚨 问题2: F-Score整体偏低 (0.648)
**可能原因**:
- 类别不平衡严重 (Class 16有16.3倍差异)
- 没有class weights
- 损失函数可能不够优化
- 训练策略不够激进

#### 🚨 问题3: 训练仍在改善 (最后50轮 +0.02)
**说明**:
- 150 epochs不够，模型未完全收敛
- 学习率下降太快 (最终只有1.6e-4)

---

## 🎯 优化策略 (按优先级)

### 优先级1: 减少过拟合 ⭐⭐⭐⭐⭐

#### 1.1 使用dropout版本模型 (已完成)
```yaml
model:
  dropout: 0.15  # 已添加
```
**预期提升**: Val F-Score +2-3% (减少过拟合)

#### 1.2 增加训练轮数
```yaml
training:
  epochs: 200  # 从150增加
  early_stopping_patience: 50  # 从30增加
```
**原因**: 当前训练趋势仍在上升
**预期提升**: +1-2%

#### 1.3 调整学习率策略
**当前问题**: LR下降太快 (1.5e-3 → 1.6e-4 in 150 epochs)

**方案A**: 提高初始LR + Warmup
```yaml
training:
  learning_rate: 1e-3  # 从1.5e-3降低
  warmup_epochs: 10    # 从5增加
  scheduler: CosineAnnealingWarmRestarts  # 周期性重启
```

**方案B**: 保守策略 (推荐)
```yaml
training:
  learning_rate: 8e-4  # 更保守
  warmup_epochs: 5
  scheduler: CosineAnnealingLR
  T_max: 180  # 180 epochs后到达最小值
```

---

### 优先级2: 处理类别不平衡 ⭐⭐⭐⭐

#### 2.1 添加Class Weights
**分析**: Class 16几乎不存在 (16.3x不平衡)

**实施步骤**:
1. 计算类别权重
```python
# In src/utils.py
def compute_class_weights(data_dir, num_classes=19):
    from glob import glob
    class_counts = np.zeros(num_classes)

    for mask_file in glob(f"{data_dir}/masks/*.png"):
        mask = np.array(Image.open(mask_file))
        for cls in range(num_classes):
            class_counts[cls] += (mask == cls).sum()

    # Inverse frequency weighting
    total = class_counts.sum()
    weights = total / (num_classes * (class_counts + 1.0))
    weights = weights / weights.mean()  # Normalize

    # Cap extreme weights
    weights = np.clip(weights, 0.1, 10.0)
    return torch.FloatTensor(weights)
```

2. 在CombinedLoss中使用
```python
class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=0.5, class_weights=None):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        ...
```

**预期提升**: +3-5% (显著改善小类别F-Score)

#### 2.2 Focal Loss (可选)
对于极度不平衡的类别，考虑Focal Loss:
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
```

**预期提升**: +1-2%

---

### 优先级3: 优化数据和训练 ⭐⭐⭐

#### 3.1 多尺度训练
**当前**: 固定512x512
**改进**: 随机尺度 [384, 448, 512, 576]

```python
# In dataset.py
class MultiScaleResize:
    def __init__(self, scales=[384, 448, 512, 576]):
        self.scales = scales

    def __call__(self, img, mask):
        scale = random.choice(self.scales)
        img = cv2.resize(img, (scale, scale))
        mask = cv2.resize(mask, (scale, scale), interpolation=cv2.INTER_NEAREST)

        # Crop to 512x512
        if scale > 512:
            x = random.randint(0, scale - 512)
            y = random.randint(0, scale - 512)
            img = img[y:y+512, x:x+512]
            mask = mask[y:y+512, x:x+512]
        elif scale < 512:
            # Pad to 512
            ...
        return img, mask
```

**预期提升**: +1-2% (更好的泛化)

#### 3.2 Mixup/CutMix (谨慎使用)
分割任务中效果不确定，但可以尝试：
```python
def cutmix(img1, mask1, img2, mask2, alpha=0.5):
    lam = np.random.beta(alpha, alpha)
    H, W = img1.shape[:2]
    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    img1[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
    mask1[y1:y2, x1:x2] = mask2[y1:y2, x1:x2]

    return img1, mask1
```

**预期提升**: +0.5-1% (可能负面)

#### 3.3 在线难样本挖掘
每个epoch后分析预测错误最严重的样本，增加其采样权重：
```python
class WeightedRandomSampler:
    # 根据上一轮的F-Score给样本加权
    # 低F-Score样本权重高
```

**预期提升**: +1-2%

---

### 优先级4: 模型架构优化 ⭐⭐

#### 4.1 增加模型深度 (在参数限制内)
**当前**: depths=[1, 2, 2, 2] (7层)
**可能**: depths=[2, 2, 3, 2] (9层)

需要减小embed_dims保持参数 <1.82M:
```python
embed_dims = [28, 56, 112, 180]  # 从[32, 64, 128, 192]减小
depths = [2, 2, 3, 2]
```

**预期提升**: +1-2% (更深的特征)

#### 4.2 添加辅助损失
在decoder中间层添加辅助监督：
```python
class MLPDecoder:
    def forward(self, ...):
        # ...
        # Add auxiliary head at stage 3
        aux_out = self.aux_head(_c3)  # Lightweight prediction
        return main_out, aux_out

# Loss
loss = main_loss + 0.4 * aux_loss
```

**预期提升**: +1-2%

---

## 📋 推荐的实施计划

### 阶段1: 立即可做 (1-2小时)

1. ✅ **添加dropout** (已完成)
2. ⚡ **计算并添加class weights** (代码已给出)
   - 预期效果最明显 (+3-5%)
3. ⚡ **调整训练配置**:
   - epochs: 200
   - learning_rate: 8e-4
   - early_stopping_patience: 50

**预期总提升**: Val F-Score 0.648 → **0.70-0.72**

### 阶段2: 如果阶段1效果不够 (2-3小时)

4. 实施多尺度训练
5. 添加Focal Loss或调整class weights
6. 在线难样本挖掘

**预期总提升**: Val F-Score 0.70-0.72 → **0.73-0.76**

### 阶段3: 最后冲刺 (需要重新设计)

7. 尝试更深的架构
8. 添加辅助损失
9. 集成学习 (训练3个模型，测试时ensemble)

**预期总提升**: Val F-Score 0.73-0.76 → **0.76-0.80**

---

## 🚀 快速开始代码

### 添加Class Weights

```python
# 1. 计算权重 (运行一次)
python -c "
from src.utils import compute_class_weights
weights = compute_class_weights('data/train')
print('Class weights:', weights.tolist())
import torch
torch.save(weights, 'data/class_weights.pt')
"

# 2. 在trainer.py中加载
class_weights = torch.load('data/class_weights.pt').to(device)
criterion = CombinedLoss(ce_weight=1.0, dice_weight=0.5, class_weights=class_weights)

# 3. 训练
python main.py --config configs/optimized.yaml
```

---

## 📊 预期最终结果

| 阶段 | 改进 | Val F-Score | 达成目标 |
|------|------|-------------|----------|
| 当前 | - | 0.648 | ❌ |
| +Dropout | 减少过拟合 | 0.66-0.68 | ❌ |
| +Class Weights | 处理不平衡 | 0.70-0.72 | ⚠️ |
| +多尺度训练 | 更好泛化 | 0.73-0.76 | ✅ |
| +架构优化 | 提升容量 | 0.76-0.80 | ✅✅ |

**保守估计**: 0.72-0.75 (满足竞赛要求)
**理想情况**: 0.76-0.78 (前25%)
**最佳情况**: 0.80+ (前10%)

---

## ⚠️ 注意事项

1. **参数限制**: 始终保持 <1,821,085 参数
2. **提交兼容性**: 确保submission/solution/与训练代码一致
3. **验证集大小**: 当前只有10% (100张)，结果可能有波动
4. **时间限制**: 10月14日截止，优先实施高优先级改进

**建议**: 先实施阶段1，训练看效果。如果达到0.70+，再考虑阶段2。
