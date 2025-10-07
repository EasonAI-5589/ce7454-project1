# 🎯 超参数调优策略 - 基于0.6753基线

## 📊 当前最佳结果

**Baseline (optimized.yaml):**
- Checkpoint: `microsegformer_20251006_132005`
- **最佳Val F-Score: 0.6753** (Epoch 124/144)
- 参数量: 1,721,939 (94.6%)

**失败的尝试:**
- LMSA版本: 0.1705 (Epoch 6) ❌ - 效果太差
- Class weights: 0.5985 (Epoch 82) ❌ - 比baseline低11%

## 🔬 三个优化方向

### ✅ 推荐：optimized_v2.yaml (激进版)
**目标: 0.72-0.74 F-Score**

```yaml
关键改动:
- LR: 8e-4 → 1e-3 (+25%)        # 加快学习速度
- Dice weight: 1.0 → 1.2         # 更关注F-Score
- Warmup: 5 → 10 epochs          # 更稳定的起步
- Epochs: 200 → 250              # 训练更久
- Augmentation: 更激进            # 更强的泛化
  - rotation: 15° → 20°
  - scale: [0.9,1.1] → [0.85,1.15]
  - brightness/contrast: 0.2 → 0.3
```

**适合场景**:
- 你有充足的GPU时间
- 想快速看到提升
- 愿意承担过拟合风险

**预期**: 前50 epochs会学得更快，最终F-Score: 0.72-0.74

---

### ✅✅ 最推荐：optimized_v3.yaml (保守版)
**目标: 0.70-0.72 F-Score**

```yaml
关键改动:
- Dropout: 0.15 → 0.12            # 降低正则化
- LR: 8e-4 → 7e-4 (-12.5%)        # 更稳定收敛
- Epochs: 200 → 300               # 充分训练
- Early stopping: 50 → 80         # 更耐心
- Augmentation: 轻微增强          # 小幅改进
```

**适合场景**:
- 稳妥提升，风险最低
- 基于0.6753稳步改进
- 时间充裕，可以训练300轮

**预期**: 收敛更稳定，最终F-Score: 0.70-0.72

---

### 🚀 optimized_v4.yaml (调度器优化)
**目标: 0.71-0.73 F-Score**

```yaml
关键改动:
- LR: 8e-4 → 1.2e-3 (+50%)        # 高起点
- Warmup: 5 → 15 epochs           # 长预热
- Epochs: 250                     # 让cosine完整衰减
- 依赖cosine scheduler充分衰减
```

**适合场景**:
- 想要前期快速学习
- 后期精细调优
- 适合调度器实验

**预期**: 前100 epochs快速收敛，后期精调

---

## 📈 参数调优对比表

| 参数 | Baseline | V2 (激进) | V3 (保守) | V4 (调度) |
|-----|----------|----------|----------|----------|
| **Learning Rate** | 8e-4 | 1e-3 | 7e-4 | 1.2e-3 |
| **Dropout** | 0.15 | 0.15 | 0.12 | 0.15 |
| **Dice Weight** | 1.0 | 1.2 | 1.0 | 1.0 |
| **Warmup** | 5 | 10 | 8 | 15 |
| **Epochs** | 200 | 250 | 300 | 250 |
| **Early Stop** | 50 | 60 | 80 | 70 |
| **Rotation** | 15° | 20° | 15° | 18° |
| **预期F-Score** | 0.6753 | 0.72-0.74 | 0.70-0.72 | 0.71-0.73 |
| **风险等级** | ✅ Low | ⚠️ Medium | ✅ Low | ⚠️ Medium |

---

## 🎲 训练建议

### 方案A：稳妥提升 (推荐)
```bash
# 在服务器上先跑V3 (最稳)
python main.py --config configs/optimized_v3.yaml --device cuda > logs/train_v3.log 2>&1 &

# 如果V3效果好(>0.70)，再跑V2冲击更高
python main.py --config configs/optimized_v2.yaml --device cuda > logs/train_v2.log 2>&1 &
```

### 方案B：快速实验
```bash
# 同时跑V2和V3对比
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/optimized_v2.yaml --device cuda > logs/train_v2.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/optimized_v3.yaml --device cuda > logs/train_v3.log 2>&1 &

# 50个epoch后看哪个更好
tail -100 logs/train_v2.log | grep "Epoch.*Summary"
tail -100 logs/train_v3.log | grep "Epoch.*Summary"
```

### 方案C：全面测试
```bash
# 三个都跑，最后选最好的
# 需要3个GPU或者分时间跑
```

---

## 🔍 监控关键指标

训练过程中重点关注：

1. **前20 epochs**: Val F-Score应该 > 0.40
2. **50 epochs**: Val F-Score应该 > 0.60
3. **100 epochs**: Val F-Score应该 > 0.68
4. **最终**: Val F-Score目标 > 0.72

如果某个版本在50 epochs时还没到0.60，可以提前停止。

---

## ⚠️ 为什么LMSA失败了？

从你的日志看：
- Epoch 2: Val F-Score 0.1185
- Epoch 3: Val F-Score 0.1225
- Epoch 5: Val F-Score 0.1712
- Epoch 6: Val F-Score 0.1705

**问题分析**:
1. ❌ LMSA模块可能引入了架构不稳定
2. ❌ 额外的参数没有充分训练
3. ❌ 可能需要更低的学习率来训练LMSA

**结论**: 暂时放弃LMSA，专注优化proven的baseline架构

---

## 📝 技术报告可以这样写

### 消融实验
```
| Configuration | Val F-Score | Notes |
|---------------|-------------|-------|
| Baseline      | 0.6753      | Proven architecture |
| +Class Weights| 0.5985      | -11.4% (failed) |
| +LMSA Module  | 0.1705      | -74.7% (failed) |
| +V3 Tuning    | 0.7X        | Conservative improvement |
| +V2 Tuning    | 0.7X        | Aggressive improvement |
```

### 超参数分析
- 学习率敏感性分析
- Dropout正则化效果
- 数据增强策略对比
- 训练时长影响

---

## ✅ 行动计划

**立即行动**:
1. 在服务器上启动 `optimized_v3.yaml` (最稳妥)
2. 同时启动 `optimized_v2.yaml` (如果有第二个GPU)
3. 监控50 epochs后的结果
4. 选择更好的继续训练

**预期结果**:
- V3应该能稳定达到 0.70-0.72
- V2可能达到 0.72-0.74 (如果不过拟合)

**目标**:
- 📌 最低目标: 0.70 (保底可以接受)
- 🎯 理想目标: 0.72-0.74 (争取达到)
- 🚀 冲刺目标: 0.75+ (如果运气好)

祝训练顺利！🎉
