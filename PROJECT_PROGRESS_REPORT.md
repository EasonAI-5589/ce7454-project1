# CE7454 Face Parsing Project - 完整进展报告

**项目截止日期:** 2025-10-14
**当前日期:** 2025-10-10
**剩余时间:** 4天

---

## 📊 整体进展概览

### 关键指标

| 指标 | 当前最佳 | 初始baseline | 提升 |
|------|----------|--------------|------|
| **Val F-Score** | **0.7041** | 0.6036 | **+16.6%** |
| **Test F-Score (Codabench)** | **0.72** | N/A | - |
| **模型参数** | 1.75M | 1.75M | 在限制内 |
| **训练轮数** | 113 epochs | 89 epochs | - |

### 实验统计
- **总实验数:** 23个
- **完成实验:** 18个 (有结果)
- **进行中/失败:** 5个
- **最佳模型:** `microsegformer_20251009_173630` (lmsa_gpu_aug)

---

## 🏆 Top 10 实验结果排名

| 排名 | 实验名称 | Val F-Score | Epoch | 关键配置 | 备注 |
|------|----------|-------------|-------|----------|------|
| 🥇 **1** | **lmsa_gpu_aug** | **0.7041** | 113/143 | LR=8e-4, Dice=1.5, LMSA✓ | **当前最佳** |
| 🥈 2 | lmsa_continue_v2 | 0.6958 | 3/33 | LR=4e-4, Dice=1.5, LMSA✓ | 继续训练 |
| 🥉 3 | lmsa_enhanced_dice | 0.6889 | 92/112 | LR=8e-4, Dice=1.5, LMSA✓ | Dice优化 |
| 4 | lmsa_dice2.5_aggressive | 0.6827 | 78/78 | LR=6e-4, Dice=2.5, LMSA✓ | 提前停止 |
| 5 | lmsa_v1 | 0.6819 | 80/100 | LR=8e-4, Dice=1.0, LMSA✓ | 首个LMSA |
| 6 | lmsa_strong_aug | 0.6769 | 75/95 | LR=8e-4, Dice=1.0, LMSA✓ | 强增强 |
| 7 | optimized_with_dropout | 0.6753 | 124/144 | LR=8e-4, Dice=1.0, LMSA✗ | 无LMSA |
| 8 | optimized_v3_conservative | 0.6736 | 128/148 | LR=7e-4, Dice=1.0, LMSA✗ | 保守LR |
| 9 | lmsa_enhanced_dice_v2 | 0.6702 | 68/98 | LR=8e-4, Dice=2.0, LMSA✓ | Dice=2.0 |
| 10 | lmsa_focal_v1 | 0.6702 | 78/98 | LR=5e-4, Dice=1.0, LMSA✓ | Focal Loss |

---

## 🔬 实验阶段总结

### Phase 1: Baseline探索 (10/5)
**目标:** 建立基线性能

| 实验 | Val F-Score | 关键发现 |
|------|-------------|----------|
| baseline_microsegformer (LR=1e-3) | N/A | 训练不稳定 |
| baseline_microsegformer (LR=2e-3) | 0.6036 | LR过高 |
| baseline_microsegformer (LR=2e-3, Dice=0.5) | 0.6483 | **首个可用baseline** |

**结论:** LR=2e-3, Dice=0.5可以工作，但性能一般

---

### Phase 2: Dropout优化 (10/6)
**目标:** 通过Dropout提高泛化

| 实验 | Val F-Score | Dropout | 发现 |
|------|-------------|---------|------|
| optimized_with_dropout (Dice=0.5) | 0.6546 | 0.15 | 轻微提升 |
| optimized_with_dropout (Dice=1.0) | 0.6753 | 0.15 | **Dice=1.0更好** |

**结论:** Dropout=0.15有效，Dice weight需要≥1.0

---

### Phase 3: Class Weighting尝试 (10/6)
**目标:** 用类别权重处理不平衡

| 实验 | Val F-Score | 结果 |
|------|-------------|------|
| class_weighted_v1 | 0.5985 | ❌ 性能下降 |

**结论:** 类别权重在这个数据集上不work，放弃

---

### Phase 4: LMSA突破 (10/7)
**目标:** 引入Lightweight Multi-Scale Attention

| 实验 | Val F-Score | 关键改进 |
|------|-------------|----------|
| lmsa_v1 | 0.6819 | **LMSA首次成功！** |
| lmsa_focal_v1 (LR=5e-4) | 0.6702 | Focal Loss效果一般 |

**结论:** LMSA是关键突破，提升~1%

---

### Phase 5: Dice Loss调优 (10/8-10/9)
**目标:** 优化Dice loss权重

| 实验 | Val F-Score | Dice Weight | 发现 |
|------|-------------|-------------|------|
| lmsa_ultimate_v1 | 0.6456 | 1.0 | Baseline |
| lmsa_enhanced_dice | 0.6889 | 1.5 | ✅ **最优** |
| lmsa_enhanced_dice_v2 | 0.6702 | 2.0 | 开始下降 |
| lmsa_dice2.5_aggressive | 0.6827 | 2.5 | 过大 |

**结论:** **Dice=1.5是最优值**

---

### Phase 6: GPU增强突破 (10/9) ⭐
**目标:** 用GPU增强提升泛化

| 实验 | Val F-Score | Test F-Score | 配置 |
|------|-------------|--------------|------|
| **lmsa_gpu_aug** | **0.7041** | **0.72** | LR=8e-4, Dice=1.5, GPU Aug✓ |

**突破点:**
- ✅ 首次突破0.70大关
- ✅ Test 0.72 (Codabench实测)
- ✅ GPU增强(Kornia)有效

---

### Phase 7: 继续训练尝试 (10/9)
**目标:** 从最佳checkpoint继续训练

| 实验 | Val F-Score | 起始点 | 结果 |
|------|-------------|--------|------|
| lmsa_continue_v2 | 0.6958 | 0.6889 | 仅+0.7% |

**结论:** 继续训练收益有限

---

### Phase 8: Warm Restarts失败 (10/9) ❌
**目标:** 用CosineAnnealingWarmRestarts提升

| 实验 | Val F-Score | Epoch | 状态 |
|------|-------------|-------|------|
| lmsa_dice1.5_warm_restart | 0.6294 | 56/200 | 被杀 |
| lmsa_dice2.0_warm_restart | 0.5877 | 40/200 | 被杀 |

**问题分析:**
- 训练不稳定，F-Score波动大
- 进程被系统杀掉
- Warm Restarts在这个任务上不适合

**结论:** CosineAnnealingLR更稳定，放弃Warm Restarts

---

### Phase 9: 高级数据增强 (10/10) 🆕
**目标:** MixUp+CutMix提高泛化，缩小Val-Test gap

| 实验 | Val F-Score | Test F-Score | 状态 |
|------|-------------|--------------|------|
| lmsa_advanced_aug | N/A | N/A | **待启动** |

**技术方案:**
- ✅ MixUp (prob=0.3, alpha=0.2)
- ✅ CutMix (prob=0.3, alpha=1.0)
- ✅ 纯PyTorch实现，全GPU
- ✅ 参数不变(1.75M)

**预期:**
- Val: 0.71-0.72
- Test: 0.73-0.75
- Gap: <1%

---

## 📈 性能提升轨迹

```
Baseline (10/5):     0.6036  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dropout优化 (10/6):  0.6753  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LMSA突破 (10/7):     0.6819  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dice调优 (10/8):     0.6889  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GPU增强 (10/9):      0.7041  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

高级增强 (10/10):    0.72?   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━? (预期)
```

**总提升:** 0.6036 → 0.7041 = **+16.6%**

---

## 🔑 关键发现总结

### ✅ 有效策略

1. **LMSA模块** - 最关键突破 (+~1%)
   - Lightweight Multi-Scale Attention
   - 专为小目标设计
   - 参数高效(~26K)

2. **Dice Loss调优** - Dice=1.5最优
   - Dice=1.0: baseline
   - Dice=1.5: +0.7% ✓
   - Dice=2.0-2.5: 过大，性能下降

3. **Dropout正则化** - Dropout=0.15有效
   - 提高泛化能力
   - 防止过拟合

4. **GPU数据增强** - Kornia增强有效
   - 提升~1.5%
   - 无CPU瓶颈

5. **学习率优化** - LR=8e-4最佳
   - 7e-4: 保守
   - 8e-4: 最优 ✓
   - 更高: 不稳定

6. **CosineAnnealingLR** - 稳定可靠
   - 比StepLR更平滑
   - 比Warm Restarts更稳定

### ❌ 无效/失败策略

1. **Class Weighting** - 性能下降
   - 可能数据分布不够不平衡
   - 或权重计算不当

2. **Focal Loss** - 效果一般
   - 轻微下降
   - 不如Dice Loss

3. **CosineAnnealingWarmRestarts** - 训练不稳定
   - F-Score波动大
   - 容易崩溃
   - 不推荐

4. **继续训练策略** - 收益有限
   - 只能提升0.5-1%
   - 不如从头训练新配置

5. **过大的Dice权重** - 性能下降
   - Dice>2.0开始负面影响
   - 可能过度强调重叠

---

## 🎯 当前最佳配置

### 模型配置
```yaml
model:
  name: microsegformer
  num_classes: 19
  use_lmsa: true         # ✓ 关键
  dropout: 0.15          # ✓ 正则化
```

### 训练配置
```yaml
training:
  epochs: 200
  optimizer: AdamW
  learning_rate: 8e-4    # ✓ 最优
  weight_decay: 1e-4
  scheduler: CosineAnnealingLR  # ✓ 稳定
  warmup_epochs: 5
  max_grad_norm: 1.0
  early_stopping_patience: 30
  use_amp: true          # ✓ 混合精度
```

### 损失配置
```yaml
loss:
  ce_weight: 1.0
  dice_weight: 1.5       # ✓ 最优
  use_focal: false
  use_class_weights: false
```

### 数据配置
```yaml
data:
  batch_size: 32
  num_workers: 4
  val_split: 0.1
```

**模型参数:** 1,747,923 (96.0% of limit)

---

## 📊 Codabench提交记录

| 版本 | Val F-Score | Test F-Score | Gap | 模型 |
|------|-------------|--------------|-----|------|
| v1 | 0.6819 | 0.72 | +5.6% | lmsa_v1 |
| v3 | 0.7041 | 0.72 | +1.6% | lmsa_gpu_aug |
| v4 (TTA) | 0.7041 | 0.72-0.73? | - | v1 + TTA |
| **v5 (待提交)** | **0.71-0.72?** | **0.73-0.75?** | **<1%?** | **advanced_aug** |

**关键问题:** Val-Test gap仍然存在
- v1: gap 5.6% (泛化好但性能低)
- v3: gap 1.6% (性能高但有过拟合)

**解决方案:** 高级数据增强(MixUp+CutMix)

---

## 🚀 下一步行动计划

### 立即执行 (今天)
1. ✅ **已完成:** 实现MixUp+CutMix GPU增强
2. ✅ **已完成:** 创建训练配置lmsa_advanced_aug.yaml
3. ✅ **已完成:** Git提交所有改动
4. ⏳ **待执行:** 在服务器启动训练

### 训练命令
```bash
# 在服务器上执行
mkdir -p logs
nohup python main.py --config configs/lmsa_advanced_aug.yaml > logs/advanced_aug.log 2>&1 &
tail -f logs/advanced_aug.log
```

### 第2天 (10/11)
1. **监控训练进度**
   - 检查Val F-Score曲线
   - 确保没有过拟合
   - 预计50-100 epochs看到效果

2. **调整策略(如需要)**
   - 如果效果不好，调整MixUp/CutMix概率
   - 或尝试更激进的增强

### 第3天 (10/12)
1. **完成训练** (预计200 epochs)
2. **生成测试集预测**
   ```bash
   python inference.py --model-path checkpoints/best_model.pth
   ```
3. **提交Codabench**
   - 目标: Test F-Score 0.73-0.75

### 第4天 (10/13)
1. **准备最终材料**
   - 技术报告(5页)
   - 代码整理
   - 打包提交

2. **应急方案**
   - 如果advanced_aug效果不好
   - 使用当前最佳(v3: 0.7041/0.72)
   - 至少保证有稳定提交

### 截止日(10/14)
1. **最终检查**
2. **NTULearn提交** (11:59PM前)

---

## 📝 技术债务/待改进

### 已识别问题
1. **边界粗糙** - 生成的mask边界不够精细
   - 原因: 解码器太简单(单次4x上采样)
   - 解决: 需要渐进上采样+skip refinement
   - 障碍: 参数预算不足(只剩73K)

2. **Val-Test gap** - 过拟合问题
   - v3: 0.7041 → 0.72 (gap 1.6%)
   - 解决中: 高级数据增强(MixUp+CutMix)

3. **小目标分割** - 眼睛、嘴巴等小区域
   - 部分解决: LMSA模块
   - 需要: 更强的数据增强(CutMix专门针对这个)

### 架构限制
- **参数限制:** 1.82M (严格)
  - 当前: 1.75M (96%)
  - 剩余: 73K (不够改decoder)

- **解码器简化:**
  - 当前: MLPDecoder (一次性4x上采样)
  - 理想: 渐进式上采样+skip refinement
  - 实际: 参数不够，无法实现

### 未尝试的策略
1. **Self-training/Pseudo-labeling** - 时间不够
2. **Knowledge Distillation** - 需要teacher model
3. **Multi-task Learning** - 参数不够
4. **Test-Time Augmentation (TTA)** - 已实现但效果有限

---

## 💡 经验教训

### 成功经验
1. **模块化改进** - LMSA单独模块易于集成
2. **渐进式调优** - 逐步优化超参数
3. **快速实验** - 用early stopping节省时间
4. **GPU优化** - GPU增强消除CPU瓶颈

### 失败教训
1. **避免过度复杂** - Class weighting反而降低性能
2. **Scheduler选择** - Warm Restarts不稳定
3. **继续训练收益低** - 不如重新训练
4. **参数预算规划** - 应该提前规划decoder改进

---

## 📊 最终目标

### 性能目标
- **Val F-Score:** 0.71-0.72 ✓
- **Test F-Score:** 0.73-0.75 (目标)
- **排名:** Top 30% (15-18分) → 争取Top 20% (19-21分)

### 提交材料
- ✅ 训练代码
- ✅ 推理代码
- ✅ 最佳模型checkpoint
- ⏳ 测试集预测(Codabench)
- ⏳ 技术报告(5页)
- ⏳ 打包提交(.zip)

---

## 📞 联系信息

- **Checkpoint最佳:** `checkpoints/microsegformer_20251009_173630/`
- **配置文件:** `configs/lmsa_advanced_aug.yaml`
- **文档:** `ADVANCED_AUG_README.md`, `START_ADVANCED_AUG.md`

**最后更新:** 2025-10-10
**状态:** 高级数据增强待启动，预计2天出结果

---

🎯 **当前目标:** 在服务器启动 `lmsa_advanced_aug` 训练，冲击Test 0.73-0.75！
