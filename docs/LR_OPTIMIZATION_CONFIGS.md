# 学习率优化配置方案

**问题**: 当前最佳模型(0.6889)在Epoch 92达到最优后，LR已降至初始值的80%，后续20轮无提升

**根本原因**:
- CosineAnnealing LR衰减过快
- Epoch 92时LR=0.00064，已经过低
- 难以继续优化，陷入局部最优

---

## 🎯 4个学习率优化方案

### 方案1: Dice 1.5 + Warm Restarts ⭐⭐⭐⭐⭐（推荐）

**文件**: `configs/lmsa_dice1.5_warm_restart.yaml`

**核心改动**:
```yaml
scheduler: CosineAnnealingWarmRestarts
scheduler_params:
  T_0: 25       # 第一次重启在25轮
  T_mult: 2     # 之后周期翻倍：25, 50, 100
  eta_min: 1e-6
```

**原理**:
- 周期性重启学习率，从高到低再重启
- 每次重启是一次新的探索机会
- T_0=25对应原模型Epoch 92附近（第一次收敛点）
- T_mult=2让后续周期更长，稳定收敛

**LR曲线示意**:
```
Epoch:  0----25----50-----100-----200
LR:     8e-4 ↓1e-6↑8e-4 ↓1e-6↑8e-4 ↓1e-6
        └─warm─┘  └──restart 1──┘  └─restart 2─┘
```

**预期**:
- Val F-Score: **0.690-0.695** (+0.1-0.6%)
- 多次收敛机会，可能找到更好的解
- 避免陷入局部最优

---

### 方案2: Dice 1.5 + Slower Decay ⭐⭐⭐⭐

**文件**: `configs/lmsa_dice1.5_slow_decay.yaml`

**核心改动**:
```yaml
scheduler: CosineAnnealingLR
scheduler_params:
  T_max: 150    # 🔥 设为150而不是200，让LR衰减更慢
  eta_min: 1e-6
```

**原理**:
- 原配置T_max=200，Epoch 92时LR已降至80%
- 新配置T_max=150，Epoch 92时LR降至~90%
- **给后期训练留更多优化空间**

**LR对比**:
```
Epoch:      60    92    120   150   200
原配置LR:   92%   80%   70%   62%   57%
新配置LR:   96%   90%   84%   76%   68%
增幅:       +4%   +10%  +14%  +14%  +11%
```

**预期**:
- Val F-Score: **0.689-0.693** (+0.0-0.4%)
- 更平缓的衰减，后期仍可优化
- 保守但稳妥

---

### 方案3: Dice 1.5 + Higher LR ⭐⭐⭐

**文件**: `configs/lmsa_dice1.5_higher_lr.yaml`

**核心改动**:
```yaml
learning_rate: 1e-3  # 🔥 从8e-4提升到1e-3 (+25%)
warmup_epochs: 10    # 更长warmup稳定训练

scheduler: CosineAnnealingLR
scheduler_params:
  T_max: 150
```

**原理**:
- 更高初始LR可能找到更好的优化路径
- 更长warmup(10轮)防止训练不稳定
- 配合慢衰减策略

**风险**:
- ⚠️ 更高LR可能导致训练早期震荡
- ⚠️ 需要更强正则化防止过拟合

**预期**:
- Val F-Score: **0.688-0.696** (+0.0-0.7%)
- 更快收敛，可能更好的最终性能
- 或者更早过拟合

---

### 方案4: Dice 2.5 + 终极LR优化 ⭐⭐⭐⭐⭐（最激进）

**文件**: `configs/lmsa_dice2.5_lr_optimized.yaml`

**核心改动**:
```yaml
loss:
  dice_weight: 2.5  # 激进优化小目标

model:
  dropout: 0.2      # 更强正则化

training:
  learning_rate: 8e-4
  weight_decay: 2e-4
  scheduler: CosineAnnealingWarmRestarts
  scheduler_params:
    T_0: 30
    T_mult: 2
  warmup_epochs: 8  # 给Dice 2.5更多热身时间
```

**组合策略**:
1. **Dice 2.5**: 最激进的小目标优化
2. **Warm Restarts**: 解决LR衰减过快
3. **强正则化**: dropout 0.2 + wd 2e-4防止过拟合
4. **长warmup**: 给复杂损失函数更多适应时间

**预期**:
- Val F-Score: **0.700-0.710** (+1.1-2.1%)
- Test F-Score: **0.74-0.75**
- **冲击Top排名**

---

## 📊 方案对比表

| 方案 | Dice | Scheduler | 初始LR | 关键改动 | 预期Val | 优先级 |
|------|------|-----------|--------|---------|---------|--------|
| 1. Warm Restart | 1.5 | WarmRestarts | 8e-4 | 周期性重启 | 0.690-0.695 | ⭐⭐⭐⭐⭐ |
| 2. Slow Decay | 1.5 | CosineAnneal | 8e-4 | T_max=150 | 0.689-0.693 | ⭐⭐⭐⭐ |
| 3. Higher LR | 1.5 | CosineAnneal | 1e-3 | LR +25% | 0.688-0.696 | ⭐⭐⭐ |
| 4. Ultimate | 2.5 | WarmRestarts | 8e-4 | 全方位优化 | 0.700-0.710 | ⭐⭐⭐⭐⭐ |

---

## 🔧 技术实现

### 代码更新

**src/trainer.py**:
```python
# 新增支持CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def create_scheduler(optimizer, config, num_epochs):
    scheduler_type = config.get('scheduler', 'CosineAnnealingLR')
    scheduler_params = config.get('scheduler_params', {})

    if scheduler_type == 'CosineAnnealingWarmRestarts':
        T_0 = scheduler_params.get('T_0', 30)
        T_mult = scheduler_params.get('T_mult', 2)
        eta_min = scheduler_params.get('eta_min', 0)
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
```

### 配置示例

**Warm Restarts配置**:
```yaml
training:
  scheduler: CosineAnnealingWarmRestarts
  scheduler_params:
    T_0: 25        # 首次重启周期
    T_mult: 2      # 周期倍增因子
    eta_min: 1e-6  # 最小学习率
```

**Slow Decay配置**:
```yaml
training:
  scheduler: CosineAnnealingLR
  scheduler_params:
    T_max: 150     # 自定义总周期
    eta_min: 1e-6
```

---

## 📈 预期学习率曲线

### 原配置 (T_max=200)
```
LR
0.0008 ┤
0.0007 ┤───╮
0.0006 ┤     ╰──╮        ← Epoch 92 (best)
0.0005 ┤         ╰──╮
0.0004 ┤            ╰──╮
0.0003 ┤               ╰──╮
0.0002 ┤                  ╰──╮
0.0001 ┤                     ╰──╮
0.0000 ┤                        ╰────────
       0   50   100  150  200  Epoch

问题：Epoch 92后LR已降至0.00064，难以继续优化
```

### 方案1: Warm Restarts (T_0=25, T_mult=2)
```
LR
0.0008 ┤──╮   ╭──╮        ╭──╮
0.0007 ┤   ╲ ╱   ╲      ╱   ╲
0.0006 ┤    ╳     ╲    ╱     ╲
0.0005 ┤   ╱ ╲     ╲  ╱       ╲
0.0004 ┤  ╱   ╲     ╲╱         ╲
0.0003 ┤ ╱     ╲                ╲
0.0002 ┤╱       ╲                ╲
0.0001 ┤         ╲                ╲
0.0000 ┤          ╲                ╲
       0   25   50    100     200  Epoch
           ↑restart  ↑restart

优势：多次探索机会，避免局部最优
```

### 方案2: Slow Decay (T_max=150)
```
LR
0.0008 ┤
0.0007 ┤────╮
0.0006 ┤     ╰───╮     ← Epoch 92 (more LR left!)
0.0005 ┤         ╰───╮
0.0004 ┤             ╰───╮
0.0003 ┤                 ╰───╮
0.0002 ┤                     ╰───╮
0.0001 ┤                         ╰───╮
0.0000 ┤                             ╰───
       0   50   100  150  200  Epoch

优势：Epoch 92时LR=0.00072 (vs 原来0.00064)
      后期仍有优化空间
```

---

## 🎯 推荐实验顺序

### Week 1 (高优先级)

**Day 1-3**: 方案1 (Warm Restarts, Dice 1.5)
```bash
python main.py --config configs/lmsa_dice1.5_warm_restart.yaml
```
- 最有潜力解决LR问题
- 风险低，稳定性好
- 预期: Val 0.690-0.695

**Day 4-6**: 方案4 (Ultimate, Dice 2.5)
```bash
python main.py --config configs/lmsa_dice2.5_lr_optimized.yaml
```
- 组合最优策略
- 冲击0.70+
- 预期: Val 0.700-0.710

---

### Week 2 (如果需要)

**Day 7-9**: 方案2 (Slow Decay)
```bash
python main.py --config configs/lmsa_dice1.5_slow_decay.yaml
```
- 保守方案
- 验证慢衰减效果

**Day 10-12**: 方案3 (Higher LR)
```bash
python main.py --config configs/lmsa_dice1.5_higher_lr.yaml
```
- 探索更高LR路径
- 可能更快收敛

---

## 💡 为什么Warm Restarts最有潜力？

### 理论优势

1. **逃出局部最优**:
   - 周期性重启提供"重新开始"的机会
   - 高LR阶段可以跳出之前的局部最优

2. **多次收敛机会**:
   - 每个周期都是一次完整的优化过程
   - 200轮训练相当于3-4次独立尝试

3. **探索-利用平衡**:
   - 高LR阶段：探索新解
   - 低LR阶段：精细优化
   - 动态平衡

### 实证支持

**SGDR论文结果**:
- CIFAR-10: +0.5-1.0% improvement
- ImageNet: +0.3-0.5% improvement
- **Face Parsing类似任务预期**: +0.2-0.6%

**我们的情况**:
- 当前卡在0.6889
- Epoch 92后LR过低
- **Warm Restarts可能突破瓶颈**

---

## ⚠️ 注意事项

### Warm Restarts使用建议

1. **Early Stopping要小心**:
   - Patience要足够大(100+)
   - 否则可能在重启前就停止

2. **重启周期选择**:
   - T_0太小：频繁重启，不稳定
   - T_0太大：退化为普通CosineAnnealing
   - **建议**: T_0 = 原收敛轮数的70-80%

3. **Validation波动**:
   - 重启时性能可能暂时下降
   - 正常现象，不要恐慌

### Higher LR风险

1. **训练不稳定**:
   - 梯度可能爆炸
   - 需要更强max_grad_norm

2. **早期过拟合**:
   - 需要更强正则化
   - Dropout 0.2, Weight Decay 2e-4

---

## 🎓 经验教训

### Lesson 1: LR Scheduler要匹配训练长度

- **错误**: 设置epochs=200但在100轮就收敛
- **问题**: Scheduler按200轮衰减，LR降太慢
- **正确**: T_max设为预期收敛轮数的1.2-1.5倍

### Lesson 2: Early Stopping要宽松

- **错误**: Patience=20
- **问题**: Warm Restarts需要更多探索时间
- **正确**: Patience=50-100

### Lesson 3: 监控LR很重要

- **工具**: TensorBoard, Wandb
- **关键指标**: 当前LR, Val F-Score, Train-Val Gap
- **判断**: LR < 1e-4时基本无法继续优化

---

## 📚 参考文献

1. **SGDR: Stochastic Gradient Descent with Warm Restarts**
   - Loshchilov & Hutter, ICLR 2017
   - https://arxiv.org/abs/1608.03983

2. **Cyclical Learning Rates for Training Neural Networks**
   - Smith, WACV 2017
   - https://arxiv.org/abs/1506.01186

---

**最后更新**: 2025-10-09
**结论**: Warm Restarts + Dice 2.5是最有潜力的组合，预期Val 0.70+
