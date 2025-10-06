# F-Score 计算说明

## 重要概念区分

### 1. F-Score ≠ Loss (它们完全不同！)

#### 📊 F-Score (评测指标)
- **用途**: 评估模型性能（越高越好）
- **范围**: 0.0 - 1.0
- **位置**: [src/utils.py:10-45](src/utils.py#L10-L45)
- **作用**:
  - 训练时监控性能
  - 验证集选择最佳模型
  - **Codabench最终评测指标**

```python
def calculate_f_score(pred, target, num_classes=19, beta=1):
    """计算F-Score - 完全匹配Codabench"""
    f_scores = []

    for class_id in np.unique(mask_gt):  # 只计算GT中的类别
        tp = np.sum((mask_gt == class_id) & (mask_pred == class_id))
        fp = np.sum((mask_gt != class_id) & (mask_pred == class_id))
        fn = np.sum((mask_gt == class_id) & (mask_pred != class_id))

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f_score = (1 + beta²) * (precision * recall) / (beta² * precision + recall + 1e-7)

        f_scores.append(f_score)

    return np.mean(f_scores)  # 所有类别的平均F-Score
```

#### 🔥 Loss Function (训练损失)
- **用途**: 训练模型（越低越好）
- **范围**: 0.0 - ∞
- **位置**: [src/utils.py:53-78](src/utils.py#L53-L78)
- **作用**:
  - 反向传播更新权重
  - 优化模型参数

```python
class CombinedLoss(nn.Module):
    """训练时的Loss = CrossEntropy + Dice"""

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)      # 交叉熵损失
        dice_loss = self.dice_loss(inputs, targets)  # Dice损失

        return ce_weight * ce_loss + dice_weight * dice_loss
```

---

## 2. 你的代码中的F-Score使用

### 在训练循环中 ([src/trainer.py:125](src/trainer.py#L125))

```python
# 训练一个batch
outputs = self.model(images)
loss = self.criterion(outputs, masks)  # 计算Loss (用于优化)

# 计算F-Score (用于监控)
with torch.no_grad():
    pred = torch.argmax(outputs, dim=1)
    f_score = calculate_f_score(pred.cpu().numpy(), masks.cpu().numpy())

# 更新统计
losses.update(loss.item(), images.size(0))
f_scores.update(f_score, 1)  # F-Score已经是平均值
```

### Loss vs F-Score 的关系

| 特性 | Loss | F-Score |
|------|------|---------|
| **作用** | 优化模型参数 | 评估模型性能 |
| **梯度** | 可微分 (有梯度) | 不可微 (无梯度) |
| **训练中** | 反向传播 | 仅监控 |
| **目标** | 最小化 | 最大化 |
| **Codabench** | 不用于评测 | **最终评测指标** |

---

## 3. 为什么不直接用F-Score作为Loss？

### ❌ F-Score不能直接做Loss的原因：

1. **不可微分**: `argmax`操作无梯度
   ```python
   pred = torch.argmax(outputs, dim=1)  # 这里梯度断了
   ```

2. **离散输出**: F-Score基于0/1预测，无法反向传播

3. **计算复杂**: 需要遍历所有类别

### ✅ 实际做法：

使用**Dice Loss作为F-Score的可微分近似**

```python
# Dice Loss ≈ 可微分的F-Score
dice_loss = 1 - (2 * intersection) / (pred + target)

# 与F-Score的关系:
# Dice coefficient = 2*TP / (2*TP + FP + FN)
# F1-Score = 2*TP / (2*TP + FP + FN)  (当precision=recall时)
```

---

## 4. 当前训练配置

### Loss配置 ([configs/optimized.yaml](configs/optimized.yaml))

```yaml
loss:
  ce_weight: 1.0    # CrossEntropy权重
  dice_weight: 0.5  # Dice Loss权重

# 实际Loss = 1.0 * CE + 0.5 * Dice
```

### 为什么用这个组合？

1. **CrossEntropy Loss**:
   - 分类准确性
   - 稳定训练

2. **Dice Loss**:
   - 类似F-Score
   - 处理类别不平衡
   - 关注IoU/重叠度

3. **组合效果**:
   - CE确保基础分类正确
   - Dice优化F-Score相关指标

---

## 5. 如何查看训练中的F-Score？

### 方法1: 训练日志

```bash
# 查看训练输出
tail -f train_dropout.log

# 输出示例:
# Epoch 1/200
#   [  0/ 29] Loss: 2.4567 F-Score: 0.3245 Acc: 0.5678
#   [ 10/ 29] Loss: 1.8234 F-Score: 0.4512 Acc: 0.6234
```

### 方法2: 历史记录

```python
import json
data = json.load(open('checkpoints/[latest]/history.json'))

print("训练F-Score:", data['train_f_score'][-1])
print("验证F-Score:", data['val_f_score'][-1])
```

### 方法3: 最佳模型

```python
# 训练器会自动保存最佳F-Score的模型
# checkpoints/[name]/best_model.pth <- Val F-Score最高的模型
```

---

## 6. 关键要点总结

### ✅ 正确理解：

1. **F-Score是评测指标**，不是训练Loss
2. **Dice Loss是可微分的F-Score近似**
3. **Codabench只看F-Score**，不看Loss
4. **训练优化Loss，验证看F-Score**

### 🎯 优化策略：

```
Low Loss + Low F-Score = 模型学歪了
Low Loss + High F-Score = 理想状态 ✓
High Loss + High F-Score = 可能过拟合
High Loss + Low F-Score = 训练失败
```

### 📊 当前状态：

- **训练Loss**: CE + 0.5*Dice (优化中)
- **F-Score计算**: 完全匹配Codabench ✓
- **目标**: Val F-Score > 0.75

---

## 7. 代码位置速查

| 功能 | 文件 | 行号 |
|------|------|------|
| F-Score计算 | src/utils.py | 10-45 |
| Loss定义 | src/utils.py | 53-78 |
| 训练中使用 | src/trainer.py | 125, 164 |
| 配置 | configs/optimized.yaml | 26-28 |

---

## 8. 常见问题

**Q: 为什么Loss下降但F-Score不升？**
A: Loss和F-Score优化目标不完全一致，可能需要调整dice_weight

**Q: 能否直接优化F-Score？**
A: 不行，F-Score不可微。但Dice Loss是很好的替代

**Q: Codabench用什么评测？**
A: 只用F-Score，计算方式与你的`calculate_f_score()`完全相同

**Q: 如何提升F-Score？**
A:
1. 添加class weights (处理不平衡)
2. 增加Dice Loss权重
3. 使用dropout (减少过拟合)
4. 更多数据增强
