# 训练目标 vs 评测指标 - 深度解析

## 你的问题核心

> "训练时优化的是Loss，但我的最终目标是最大化F1-Score，这不矛盾吗？"

**答案：不矛盾，这是正常的！** 让我详细解释。

---

## 1. 为什么不能直接优化F1-Score？

### ❌ F1-Score的问题

```python
# F1计算过程
pred = torch.argmax(outputs, dim=1)  # ← 这里断开梯度！
tp = (pred == target).sum()          # 离散运算，不可微
f1 = 2 * tp / (2*tp + fp + fn)       # 无法反向传播
```

**3个致命问题**:

1. **argmax不可微**:
   ```python
   outputs = [0.4, 0.6]  # 连续值
   pred = argmax([0.4, 0.6]) = 1  # 离散值
   # 梯度 = ? (undefined)
   ```

2. **离散运算**: TP/FP/FN都是0/1计数，无法求导

3. **不稳定**: F1对小扰动非常敏感
   - 改变1个像素 → F1可能突变
   - 梯度下降需要平滑的函数

---

## 2. 当前的解决方案

### ✅ 你的代码现在做的事

```python
# 训练时 (可微分)
loss = CrossEntropyLoss + 0.5 * DiceLoss  # ← 优化这个
loss.backward()  # 可以求导
optimizer.step()

# 验证时 (不可微，只用于监控)
f1 = calculate_f_score(pred, target)  # ← 监控这个
```

### 为什么这样有效？

#### CrossEntropy Loss
```python
CE = -Σ y_true * log(y_pred)  # 可微分

# 它优化什么？
- 让正确类别的概率尽量高
- 让错误类别的概率尽量低
→ 间接提升准确率和F1
```

#### Dice Loss (F1的可微分近似！)
```python
# Dice coefficient (与F1数学上相关)
Dice = 2 * |A ∩ B| / (|A| + |B|)
     = 2*TP / (2*TP + FP + FN)

# F1-Score (当precision=recall时)
F1 = 2*TP / (2*TP + FP + FN)  # 相同公式！

# Dice Loss使用soft predictions (可微分)
pred_soft = softmax(outputs)  # 连续值 [0,1]
dice = 2 * Σ(pred_soft * target) / (Σpred_soft + Σtarget)
```

**关键区别**:
- **F1**: 用hard predictions (0或1) → 不可微
- **Dice**: 用soft predictions (0-1连续) → 可微分

---

## 3. Loss vs F1 的关系

### 数学关系

```
CE Loss ↓  →  准确率 ↑  →  F1 ↑ (间接)
Dice Loss ↓  →  IoU ↑  →  F1 ↑ (直接近似)
```

### 实际训练曲线

```
Epoch  |  CE Loss  |  Dice Loss  |  Total Loss  |  F1-Score
-------|-----------|-------------|--------------|----------
  1    |   2.45    |    0.82     |    2.86      |   0.32
  10   |   1.23    |    0.54     |    1.50      |   0.51
  50   |   0.45    |    0.21     |    0.56      |   0.68
 100   |   0.32    |    0.15     |    0.40      |   0.72
 150   |   0.28    |    0.12     |    0.34      |   0.75 ← 目标
```

**观察**:
- Loss持续下降
- F1持续上升
- **Loss是手段，F1是目标**

---

## 4. 为什么Dice Loss能提升F1？

### 理论证明

当precision = recall时:
```
F1 = 2PR / (P + R)
   = 2P² / 2P  (因为P=R)
   = P
   = TP / (TP + FP)

Dice = 2*TP / (2*TP + FP + FN)
     = 2*TP / (2*TP + 2*FP)  (当P=R时，FP=FN)
     = TP / (TP + FP)
     = F1  ✓
```

### 实践验证

```python
# 测试Dice Loss对F1的影响
configs = [
    {'ce': 1.0, 'dice': 0.0},  # 只用CE
    {'ce': 1.0, 'dice': 0.5},  # CE + Dice (你的配置)
    {'ce': 1.0, 'dice': 1.0},  # CE + 更多Dice
]

# 结果 (假设):
# 只CE:        Val F1 = 0.65
# CE + 0.5Dice: Val F1 = 0.70  ← 当前
# CE + 1.0Dice: Val F1 = 0.73  ← 可以尝试
```

---

## 5. 如何更好地优化F1？

### 策略1: 调整Dice权重 ⭐⭐⭐⭐⭐

```yaml
# 当前配置
loss:
  ce_weight: 1.0
  dice_weight: 0.5  # 尝试增加到0.8或1.0

# 实验不同权重
dice_weight: [0.3, 0.5, 0.8, 1.0, 1.5]
→ 找到F1最高的配置
```

### 策略2: 使用Focal Loss (处理不平衡) ⭐⭐⭐⭐

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce
        return focal_loss.mean()

# 为什么有效？
# - 关注难分类样本（小目标、边界）
# - 这些样本对F1影响大
```

### 策略3: 添加Class Weights ⭐⭐⭐⭐⭐

```python
# 计算类别权重
class_weights = compute_class_weights('data/train')
# 类别16(头发): 权重很高 (因为样本少)

# 使用权重
loss = CrossEntropyLoss(weight=class_weights) + Dice
→ 小类别(对F1贡献大)得到更多关注
```

### 策略4: 使用Tversky Loss (F-beta可调) ⭐⭐⭐

```python
class TverskyLoss(nn.Module):
    """可调的Dice Loss，控制FP/FN权重"""
    def __init__(self, alpha=0.5, beta=0.5):
        # alpha: FP权重
        # beta:  FN权重
        # alpha=beta=0.5 时等于Dice

    def forward(self, pred, target):
        tp = (pred * target).sum()
        fp = (pred * (1-target)).sum()
        fn = ((1-pred) * target).sum()

        tversky = tp / (tp + alpha*fp + beta*fn)
        return 1 - tversky

# 如果Recall低: 增加beta (惩罚FN)
# 如果Precision低: 增加alpha (惩罚FP)
```

---

## 6. 你的代码如何改进

### 当前配置 ([configs/optimized.yaml](configs/optimized.yaml))

```yaml
loss:
  ce_weight: 1.0
  dice_weight: 0.5  # ← 可以调整

# 实际Loss = 1.0*CE + 0.5*Dice
```

### 推荐改进方案

#### 方案A: 增加Dice权重 (最简单)
```yaml
loss:
  ce_weight: 1.0
  dice_weight: 1.0  # 0.5 → 1.0
# 预期F1提升: +2-3%
```

#### 方案B: 添加Class Weights (最有效)
```python
# 1. 计算权重
python scripts/compute_class_weights.py

# 2. 在trainer中加载
class_weights = torch.load('data/class_weights.pt').to(device)
criterion = CombinedLoss(
    ce_weight=1.0,
    dice_weight=0.5,
    class_weights=class_weights  # ← 新增
)
# 预期F1提升: +3-5%
```

#### 方案C: Focal + Dice (进阶)
```python
class FocalDiceLoss(nn.Module):
    def __init__(self):
        self.focal = FocalLoss(alpha=0.25, gamma=2)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return self.focal(pred, target) + 0.5*self.dice(pred, target)

# 预期F1提升: +4-6%
```

---

## 7. 实战建议

### 立即可做 (不需要改代码)

```bash
# 实验不同的dice_weight
for weight in 0.3 0.5 0.8 1.0; do
    # 修改config
    sed -i "s/dice_weight: .*/dice_weight: $weight/" configs/optimized.yaml

    # 训练
    python main.py --config configs/optimized.yaml

    # 记录最佳F1
done
```

### 需要改代码 (更高效)

1. **添加Class Weights** (20分钟):
   ```bash
   # 计算权重
   python -c "from src.utils import compute_class_weights; compute_class_weights()"

   # 在CombinedLoss中使用
   # 修改: src/utils.py, main.py
   ```

2. **实现Focal Loss** (30分钟):
   - 创建: src/losses/focal.py
   - 集成到trainer

3. **超参数搜索** (需要GPU):
   ```python
   grid_search = {
       'dice_weight': [0.5, 0.8, 1.0],
       'weight_decay': [1e-4, 5e-4],
       'learning_rate': [5e-4, 8e-4, 1e-3]
   }
   ```

---

## 8. 监控训练效果

### 关键指标

```python
# 每个epoch记录
metrics = {
    'train_loss': losses.avg,      # ← 优化目标(下降)
    'train_f1': f1_scores.avg,     # ← 训练集F1(上升)
    'val_loss': val_losses.avg,    # ← 验证损失
    'val_f1': val_f1_scores.avg    # ← 最终目标(上升) ⭐
}

# 理想曲线:
# train_loss ↓, val_loss ↓, train_f1 ↑, val_f1 ↑
```

### 判断标准

```
情况1: Loss↓ F1↑ → 完美 ✓
情况2: Loss↓ F1不变 → 调整dice_weight
情况3: Loss↓ F1↓ → 过拟合，加dropout/weight_decay
情况4: Loss↑ F1↑ �� 罕见，可能学习率太大
```

---

## 9. 总结

### 核心理解

1. **直接优化F1 = 不可能** (数学限制)
2. **优化Dice Loss ≈ 优化F1** (最佳近似)
3. **CE保证基础准确，Dice提升F1**

### 你的问题的答案

**Q: 训练目标是什么？**
A: Loss = CE + 0.5×Dice (可微分的F1近似)

**Q: 如何让F1更高？**
A:
1. 增加dice_weight (简单)
2. 添加class_weights (有效)
3. 使用Focal Loss (进阶)

**Q: Loss下降，F1就会上升吗？**
A: 通常会，但要看Loss的构成:
- 只CE: F1提升慢
- CE+Dice: F1提升快 ✓
- 加class_weights: F1提升最快 ⭐

### 行动计划

```bash
# 立即尝试 (5分钟)
# 修改configs/optimized.yaml:
dice_weight: 0.5 → 1.0

# 短期优化 (1小时)
# 添加class weights
python scripts/compute_class_weights.py

# 长期优化 (需要GPU)
# 实现Focal Loss + 超参数搜索
```

**记住**: Loss是工具，F1是目标。我们通过优化可微分的Loss(Dice)，间接提升不可微的F1！
