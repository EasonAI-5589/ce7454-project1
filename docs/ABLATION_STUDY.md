# 消融实验设计 (Ablation Study)

## 🎯 目标

系统地评估每个模块对模型性能的贡献,找出:
1. **哪些模块是必须的** (移除后性能大幅下降)
2. **哪些模块是有益的** (带来明显提升)
3. **哪些模块是可选的** (影响不大或负面)

## 📋 实验设计原则

### 控制变量法
- 每次只改变一个变量
- 其他配置保持完全一致
- 使用相同的数据划分(seed=42)
- 训练相同的epochs(200)

### 评估指标
- **主要指标**: Validation F-Score (class-averaged)
- **次要指标**: Test F-Score (用于最终验证)
- **参考指标**: Train F-Score, Accuracy (观察过拟合)

### 实验命名规范
```
ablation_{module}_{variant}
例如: ablation_lmsa_disabled, ablation_dice_1.0
```

---

## 🧪 实验组设计

### Baseline: 最简配置
```yaml
experiment:
  name: ablation_baseline
  
model:
  use_lmsa: false  # 不用LMSA
  dropout: 0.0     # 不用dropout
  
loss:
  ce_weight: 1.0
  dice_weight: 0.0  # 只用CE Loss
  
training:
  learning_rate: 1e-3
  weight_decay: 0.0  # 不用正则化
  scheduler: None    # 不用scheduler
  use_amp: false     # 不用混合精度
  
augmentation:
  # 不用数据增强
  horizontal_flip: 0.0
```

**预期**: Val F-Score ~0.60 (作为对比基准)

---

## 📊 系统性消融实验

### 实验组1: 模型架构 (Architecture)

| Exp ID | 配置 | LMSA | Dropout | 预期F-Score | 说明 |
|--------|------|------|---------|-------------|------|
| A0 | Baseline | ❌ | 0.0 | 0.60 | 基准 |
| A1 | +LMSA | ✅ | 0.0 | **0.64** | LMSA贡献 |
| A2 | +Dropout | ❌ | 0.15 | 0.62 | Dropout贡献 |
| A3 | Full | ✅ | 0.15 | **0.65** | 完整架构 |

**配置文件:**
```bash
configs/ablation/arch_baseline.yaml      # A0
configs/ablation/arch_lmsa_only.yaml     # A1
configs/ablation/arch_dropout_only.yaml  # A2
configs/ablation/arch_full.yaml          # A3
```

**预期发现:**
- LMSA贡献 ~4% F-Score
- Dropout贡献 ~2% F-Score
- 组合效果 ~5% F-Score

---

### 实验组2: Loss函数 (Loss Function) ⭐ **关键**

| Exp ID | CE | Dice | Focal | 预期F-Score | 说明 |
|--------|----|----|-------|-------------|------|
| L0 | 1.0 | 0.0 | ❌ | 0.60 | 只用CE |
| L1 | 1.0 | **0.5** | ❌ | 0.64 | +轻量Dice |
| L2 | 1.0 | **1.0** | ❌ | **0.68** | +标准Dice |
| L3 | 1.0 | **1.5** | ❌ | **0.69** | +强Dice |
| L4 | 1.0 | **2.0** | ❌ | **0.70+** | +超强Dice |
| L5 | 1.0 | **2.5** | ❌ | 0.71? | +极强Dice |
| L6 | 1.0 | 1.0 | ✅ | 0.66 | +Focal (负面?) |

**配置文件:**
```bash
configs/ablation/loss_ce_only.yaml          # L0
configs/ablation/loss_dice_0.5.yaml         # L1
configs/ablation/loss_dice_1.0.yaml         # L2
configs/ablation/loss_dice_1.5.yaml         # L3 (当前最佳)
configs/ablation/loss_dice_2.0.yaml         # L4 (测试中)
configs/ablation/loss_dice_2.5.yaml         # L5
configs/ablation/loss_focal.yaml            # L6
```

**预期发现:**
- CE alone: 0.60 (baseline)
- Dice 0.5: +4% (有效)
- Dice 1.0: +8% (很有效)
- Dice 1.5: +9% (**当前最佳**)
- Dice 2.0: +10%? (待验证)
- Dice 2.5: +11%? 或过头?
- Focal: -2% (负面,验证过)

**关键问题:**
- Dice weight的最优值是多少?
- 是否存在拐点?

---

### 实验组3: 优化器配置 (Optimizer)

| Exp ID | LR | Weight Decay | Scheduler | 预期F-Score | 说明 |
|--------|----|----|-----------|-------------|------|
| O0 | 1e-3 | 0.0 | None | 0.65 | 无正则化 |
| O1 | 8e-4 | 0.0 | None | 0.66 | 调整LR |
| O2 | 8e-4 | **1e-4** | None | 0.67 | +正则化 |
| O3 | 8e-4 | 1e-4 | **Cosine** | **0.68** | +调度器 |
| O4 | 8e-4 | **2e-4** | Cosine | 0.68? | 更强正则 |

**配置文件:**
```bash
configs/ablation/opt_baseline.yaml
configs/ablation/opt_lr_tuned.yaml
configs/ablation/opt_wd.yaml
configs/ablation/opt_full.yaml
configs/ablation/opt_strong_wd.yaml
```

**预期发现:**
- Weight decay贡献 ~1-2%
- Cosine scheduler贡献 ~1%
- LR调整贡献 ~1%

---

### 实验组4: 数据增强 (Augmentation)

| Exp ID | H-Flip | Rotation | Color | Strong | 预期F-Score | 说明 |
|--------|--------|----------|-------|--------|-------------|------|
| D0 | ❌ | ❌ | ❌ | ❌ | 0.66 | 无增强 |
| D1 | ✅ | ❌ | ❌ | ❌ | 0.67 | 只翻转 |
| D2 | ✅ | ✅ | ❌ | ❌ | **0.68** | +旋转 |
| D3 | ✅ | ✅ | ✅ | ❌ | **0.68** | +颜色 |
| D4 | ✅ | ✅ | ✅ | ✅ | 0.67? | +强增强(可能过头) |

**配置文件:**
```bash
configs/ablation/aug_none.yaml
configs/ablation/aug_flip_only.yaml
configs/ablation/aug_standard.yaml
configs/ablation/aug_full.yaml
configs/ablation/aug_strong.yaml
```

**预期发现:**
- Horizontal flip: +1% (cheap and effective)
- Rotation: +1% (moderate cost, effective)
- Color jitter: 0% (不太有用?)
- Strong aug: -1% (可能负面)

---

### 实验组5: GPU增强 vs CPU增强

| Exp ID | 增强位置 | num_workers | 预期速度 | 预期F-Score | 说明 |
|--------|----------|-------------|----------|-------------|------|
| G0 | CPU | 16 | 180s/epoch | 0.68 | 标准CPU |
| G1 | CPU | 8 | 190s/epoch | 0.68 | 优化CPU |
| G2 | CPU | 4 | 210s/epoch | 0.68 | 最少CPU |
| G3 | **GPU** | 4 | **150s/epoch** | **0.68** | GPU增强 |

**配置文件:**
```bash
configs/ablation/gpu_cpu_16w.yaml
configs/ablation/gpu_cpu_8w.yaml
configs/ablation/gpu_cpu_4w.yaml
configs/ablation/gpu_gpu_aug.yaml
```

**预期发现:**
- GPU增强: 速度+20%, F-Score持平
- num_workers: 对F-Score无影响,只影响速度

---

### 实验组6: 完整组合 (Full Combination)

| Exp ID | 配置描述 | 预期F-Score | 说明 |
|--------|----------|-------------|------|
| F0 | Baseline (最简) | 0.60 | 起点 |
| F1 | +LMSA | 0.64 | +架构 |
| F2 | +Dice 1.5 | 0.68 | +Loss |
| F3 | +Optimizer | 0.69 | +优化器 |
| F4 | +Augmentation | **0.70** | +增强 |
| F5 | +GPU Optimization | **0.70** | +GPU(速度) |

**累积贡献图:**
```
F-Score
0.70 ┤                           ████ F5
0.69 ┤                      ████      F4
0.68 ┤                 ████           F3
0.64 ┤            ████                F2
0.60 ┤       ████                     F1
0.60 ┤  ████                          F0
     └────┴────┴────┴────┴────┴────
        Base LMSA Loss Opt  Aug  GPU
```

---

## 🎯 关键研究问题

### Q1: Dice weight的最优值
**假设**: 存在最优点,过低或过高都不好

**实验**: L0-L6 (Dice 0.0, 0.5, 1.0, 1.5, 2.0, 2.5)

**预期结果**:
```
F-Score
0.71 ┤        ╭─╮
0.70 ┤      ╭╯  ╰╮
0.69 ┤    ╭╯     ╰╮
0.68 ┤  ╭╯        ╰╮
0.64 ┤╭╯            ╰╮
0.60 ┤╯              ╰
     └────┴────┴────┴
     0.0  1.0  2.0  3.0
         Dice Weight
```

**关键**: 找到峰值点 (可能在2.0-2.5之间)

---

### Q2: LMSA模块的真实贡献
**假设**: LMSA对小目标有帮助

**实验**: 
- A0 vs A1: 测量LMSA总贡献
- 分析: 哪些类别提升最多?

**验证方法**:
```python
# 对比各类别F-Score
baseline = [f1_class0, f1_class1, ...]
with_lmsa = [f1_class0, f1_class1, ...]

improvement = with_lmsa - baseline
# 预期: 小目标(眼睛、嘴巴)提升最大
```

---

### Q3: Focal Loss为什么不work?
**假设**: Focal Loss对这个任务不适合

**实验**: L6 vs L2

**分析**:
- 是否因为类别不平衡类型不对?
- 是否超参数(alpha, gamma)需要调整?
- 是否与Dice Loss冲突?

**进一步实验**:
```yaml
# L6a: Focal alpha=0.25, gamma=2.0
# L6b: Focal alpha=0.5, gamma=2.0
# L6c: Focal alpha=0.25, gamma=1.0
```

---

## 📝 实验执行计划

### Phase 1: 核心模块 (优先级⭐⭐⭐)
**目标**: 找到最重要的模块

```bash
# 1. Loss ablation (最关键)
python main.py --config configs/ablation/loss_ce_only.yaml      # L0
python main.py --config configs/ablation/loss_dice_1.0.yaml     # L2  
python main.py --config configs/ablation/loss_dice_1.5.yaml     # L3 (已完成)
python main.py --config configs/ablation/loss_dice_2.0.yaml     # L4 (进行中)
python main.py --config configs/ablation/loss_dice_2.5.yaml     # L5

# 2. Architecture ablation
python main.py --config configs/ablation/arch_baseline.yaml     # A0
python main.py --config configs/ablation/arch_lmsa_only.yaml    # A1
```

**预计时间**: 6个实验 × 4小时 = 24小时

**重要性**: ⭐⭐⭐⭐⭐
- 直接影响F-Score
- 找到最优Dice weight

---

### Phase 2: 优化细节 (优先级⭐⭐)
**目标**: 优化训练过程

```bash
# Optimizer ablation
python main.py --config configs/ablation/opt_baseline.yaml
python main.py --config configs/ablation/opt_full.yaml

# Augmentation ablation  
python main.py --config configs/ablation/aug_none.yaml
python main.py --config configs/ablation/aug_standard.yaml
```

**预计时间**: 4个实验 × 4小时 = 16小时

**重要性**: ⭐⭐⭐
- 影响1-2% F-Score
- 防止过拟合

---

### Phase 3: 工程优化 (优先级⭐)
**目标**: 提升训练效率

```bash
# GPU optimization
python main.py --config configs/ablation/gpu_cpu_16w.yaml
python main.py --config configs/ablation/gpu_gpu_aug.yaml
```

**预计时间**: 2个实验 × 4小时 = 8小时

**重要性**: ⭐⭐
- 不影响F-Score
- 提升训练速度20-30%

---

## 📊 结果记录模板

### 实验记录表

| Exp ID | Config | Val F-Score | Test F-Score | Train Time | GPU Util | Notes |
|--------|--------|-------------|--------------|------------|----------|-------|
| baseline | - | 0.600 | - | 6h | 65% | 基准 |
| A1 | +LMSA | 0.640 | - | 6h | 65% | +4% |
| L2 | +Dice1.0 | 0.680 | 0.72 | 6h | 65% | +8% |
| L3 | +Dice1.5 | **0.689** | **0.73** | 6h | 65% | 当前最佳 |
| L4 | +Dice2.0 | 0.70? | - | 4.5h | 90% | 测试中 |
| ... | ... | ... | ... | ... | ... | ... |

### 可视化结果

生成图表:
```python
# 1. 各模块贡献条形图
# 2. Dice weight vs F-Score曲线
# 3. 累积贡献堆叠图
# 4. 训练曲线对比
```

---

## 🎯 预期结论

### 关键发现 (预测)

1. **Loss配置是最重要的** (预计贡献50%)
   - Dice weight从0.0到1.5: +9% F-Score
   - 最优Dice weight: 2.0-2.5

2. **LMSA架构是第二重要的** (预计贡献30%)
   - 对小目标提升明显
   - +4% F-Score

3. **优化器和增强是锦上添花** (预计贡献15%)
   - Weight decay: +1-2%
   - Augmentation: +1-2%

4. **GPU优化不影响性能** (预计贡献5%)
   - 只影响速度
   - 但能节省25%训练时间

### 模块重要性排序

```
1. ⭐⭐⭐⭐⭐ Loss Function (Dice weight)
   - 贡献: +9-11% F-Score
   - 性价比: 极高 (只改配置)
   
2. ⭐⭐⭐⭐ Architecture (LMSA)
   - 贡献: +4% F-Score  
   - 性价比: 高 (已实现)
   
3. ⭐⭐⭐ Regularization (WD, Dropout)
   - 贡献: +2-3% F-Score
   - 性价比: 中等
   
4. ⭐⭐ Augmentation
   - 贡献: +1-2% F-Score
   - 性价比: 中等
   
5. ⭐ GPU Optimization
   - 贡献: 0% F-Score, +20% speed
   - 性价比: 高 (时间就是金钱)
```

---

## 📋 配置文件生成

所有消融实验配置将自动生成在:
```
configs/ablation/
├── loss_ce_only.yaml
├── loss_dice_0.5.yaml
├── loss_dice_1.0.yaml
├── loss_dice_1.5.yaml
├── loss_dice_2.0.yaml
├── loss_dice_2.5.yaml
├── arch_baseline.yaml
├── arch_lmsa_only.yaml
├── opt_baseline.yaml
├── opt_full.yaml
├── aug_none.yaml
├── aug_standard.yaml
└── gpu_gpu_aug.yaml
```

使用脚本批量生成:
```bash
python scripts/generate_ablation_configs.py
```

---

## 💰 成本估算

### 时间成本
- Phase 1 (核心): 24小时
- Phase 2 (优化): 16小时
- Phase 3 (工程): 8小时
- **总计**: 48小时 (2天全天GPU)

### GPU成本
- A100: ~$3/hour
- **总成本**: ~$150

### 性价比分析
- **必做**: Phase 1 (Loss ablation)
  - 成本: $75
  - 收益: 找到最优配置,+1-2% F-Score
  
- **推荐**: Phase 1 + Phase 2
  - 成本: $120
  - 收益: 完整优化,Technical Report素材
  
- **可选**: Phase 3
  - 成本: $24
  - 收益: 提速,但不影响性能

---

## 🎓 Technical Report素材

消融实验结果可以直接用于报告:

### Section 4: Ablation Study

**4.1 Loss Function Analysis**
- Table: Dice weight vs Performance
- Figure: Dice weight curve
- Finding: Optimal Dice weight = 2.0

**4.2 Architecture Components**
- Table: LMSA contribution
- Figure: Per-class improvement
- Finding: LMSA improves small objects

**4.3 Training Strategy**
- Table: Optimizer ablation
- Finding: Weight decay + Cosine scheduler

**4.4 Summary**
- Figure: Component contributions
- Table: Full ablation results

---

## ✅ 评估标准

### 这个消融实验是否值得做?

**YES, 如果:**
- ✅ 需要写Technical Report (必须有ablation study)
- ✅ 想要理解各模块贡献
- ✅ 想要找到最优配置
- ✅ 有充足GPU时间(2天)

**NO, 如果:**
- ❌ 只关心最终分数(直接用最佳配置)
- ❌ GPU时间紧张
- ❌ 不需要写详细报告

### 最小化方案

如果时间紧张,至少做:
```bash
# 1. Dice weight sweep (6个实验)
L0, L1, L2, L3, L4, L5

# 2. LMSA ablation (2个实验)
A0, A1
```

**最小成本**: 8个实验 × 4小时 = 32小时 = $100

**收益**:
- 找到最优Dice weight
- 验证LMSA贡献
- 足够写报告

---

## 🚀 开始执行

### 立即开始

```bash
# 1. 生成所有配置
python scripts/generate_ablation_configs.py

# 2. 开始Phase 1
python main.py --config configs/ablation/loss_dice_2.0.yaml  # 已在进行
python main.py --config configs/ablation/loss_dice_2.5.yaml  # 下一个
```

### 结果跟踪

创建实验跟踪表格:
```bash
# 使用wandb或tensorboard
wandb init
wandb agent ablation_sweep
```

---

**问题**: 这个消融实验方案是否满足你的需求?需要调整吗?
