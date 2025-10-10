# 文档重组方案

## 🎯 目标
将71个分散的Markdown文档整理成清晰的结构，提高可维护性。

---

## 📂 新的文档结构

```
ce7454-project1/
├── README.md                          # 主入口 (保留)
├── QUICK_START.md                     # 快速开始 (合并5个快速开始文档)
│
├── docs/
│   ├── README.md                      # 文档导航
│   │
│   ├── 01_setup/                      # 环境配置
│   │   ├── installation.md
│   │   └── gpu_optimization.md       # 从 GPU_OPTIMIZATION.md
│   │
│   ├── 02_training/                   # 训练指南
│   │   ├── basic_training.md         # 从 TRAINING_GUIDE.md
│   │   ├── continue_training.md      # 从 CONTINUE_TRAINING.md
│   │   └── hyperparameters.md        # 合并超参数文档
│   │
│   ├── 03_evaluation/                 # 评估与测试
│   │   ├── metrics_explanation.md    # 从 F_SCORE_EXPLANATION.md
│   │   └── model_evaluation.md
│   │
│   ├── 04_submission/                 # 提交指南
│   │   ├── codabench_guide.md        # 从 CODABENCH_README.md
│   │   └── submission_workflow.md    # 从 SUBMISSION_WORKFLOW.md
│   │
│   ├── 05_architecture/               # 模型架构
│   │   ├── microsegformer.md         # 从 architecture_overview.md
│   │   ├── model_comparison.md       # 合并模型对比文档
│   │   └── ablation_study.md         # 从 ABLATION_STUDY.md
│   │
│   └── 06_experiments/                # 实验记录 (归档)
│       ├── README.md                  # 实验索引
│       ├── training_history.md        # 合并训练分析文档
│       ├── hyperparameter_tuning.md   # 合并超参数实验
│       └── breakthrough_analysis.md   # 重要突破记录
│
└── archive/                           # 废弃文档 (不删除，备份)
    ├── old_status_reports/
    ├── deprecated_guides/
    └── experiment_logs/
```

---

## 🔄 文档整合计划

### Phase 1: 合并快速开始文档 ✅

**目标文件**: `QUICK_START.md`

**合并以下文档**:
- ✅ QUICK_START.md (保留为基础)
- ❌ QUICKSTART_CONTINUE.md → 整合为"继续训练"章节
- ❌ QUICKSTART_GPU_OPT.md → 整合为"GPU优化"章节
- ❌ FOCAL_LOSS_QUICK_START.md → 整合为"损失函数配置"章节
- ❌ START_ADVANCED_AUG.md → 整合为"数据增强"章节

**操作**:
```bash
# 1. 编辑 QUICK_START.md，添加所有内容
# 2. 移动旧文件到 archive/
mv QUICKSTART_*.md archive/deprecated_guides/
mv FOCAL_LOSS_QUICK_START.md archive/deprecated_guides/
mv START_ADVANCED_AUG.md archive/deprecated_guides/
```

---

### Phase 2: 整理实验文档 🔄

**目标**: `docs/06_experiments/`

**整合以下文档**:

**训练历史** → `training_history.md`:
- EXPERIMENT_ANALYSIS.md
- EXPERIMENT_SUMMARY.md
- TRAINING_RESULTS_SUMMARY.md
- COMPLETE_RESULTS_ANALYSIS.md
- EARLY_STOPPING_ANALYSIS.md
- WHY_LATE_TRAINING_STALLS.md
- WARM_RESTARTS_FAILURE_ANALYSIS.md
- BREAKTHROUGH_0.7041_ANALYSIS.md

**超参数调优** → `hyperparameter_tuning.md`:
- HYPERPARAMETER_OPTIMIZATION.md
- LR_OPTIMIZATION_CONFIGS.md
- LR_OVERFITTING_RELATIONSHIP.md
- DICE_WEIGHT_ANALYSIS.md
- RECOMMENDED_EXPERIMENTS.md

**焦点实验** → `focal_loss_experiments.md`:
- FOCAL_LOSS_EXPERIMENT.md

---

### Phase 3: 清理策略文档 🗑️

**这些文档已过时，可以归档**:
- OPTIMIZATION_PLAN.md → archive/
- OPTIMIZATION_STRATEGY.md → archive/
- TUNING_STRATEGY.md → archive/
- TRAINING_IMPROVEMENTS.md → archive/
- ARCHITECTURE_IMPROVEMENTS.md → archive/
- PROJECT_STATUS.md → archive/
- STATUS.md → archive/
- FINAL_CHECK_REPORT.md → archive/

---

### Phase 4: 重组技术文档 📚

**模型架构** → `docs/05_architecture/`:

**`microsegformer.md`** (主文档):
- 从 architecture_overview.md 整合

**`model_comparison.md`**:
- 合并 comparison_with_lightweight_models.md
- 合并 vit_vs_segformer_comparison.md

**`ablation_study.md`**:
- 从 docs/ABLATION_STUDY.md 移动

---

### Phase 5: 统一训练指南 📖

**`docs/02_training/`**:

**`basic_training.md`**:
- 从 TRAINING_GUIDE.md
- 添加 TRAINING_OBJECTIVE_EXPLANATION.md 内容

**`continue_training.md`**:
- 从 CONTINUE_TRAINING.md

**`hyperparameters.md`** (精简版，去掉实验记录):
- 从实验文档中提取最佳实践

**`advanced_augmentation.md`**:
- 从 ADVANCED_AUG_README.md

---

### Phase 6: 完善提交文档 📦

**`docs/04_submission/`**:

**`codabench_guide.md`**:
- 从 CODABENCH_README.md

**`submission_workflow.md`**:
- 从 SUBMISSION_WORKFLOW.md
- 整合实际提交经验

---

## 📋 执行清单

### 立即执行 (高优先级)

- [ ] 创建 `archive/` 目录结构
- [ ] 合并5个快速开始文档 → `QUICK_START.md`
- [ ] 移动过时的状态/策略文档到 `archive/`
- [ ] 创建 `docs/` 新的子目录结构
- [ ] 创建 `docs/README.md` 作为文档导航

### 中期执行 (整理实验)

- [ ] 整理实验分析文档 → `docs/06_experiments/`
- [ ] 合并超参数优化文档
- [ ] 保留关键突破分析

### 最终执行 (技术文档)

- [ ] 重组模型架构文档
- [ ] 统一训练指南
- [ ] 完善提交文档
- [ ] 更新所有文档间的链接

---

## ⚠️ 保留原则

**不要删除这些文件**:
- ✅ README.md (主入口)
- ✅ report/ 目录下所有文件 (技术报告)
- ✅ 最近的实验记录 (10月的文档)
- ✅ .claude/CLAUDE.md (项目指令)

**可以归档** (移到 archive/):
- 9月的计划文档
- 重复的状态报告
- 过时的策略文档

**需要合并** (去重):
- 多个快速开始文档 → 1个
- 多个超参数文档 → 1个总结
- 多个训练分析 → 按时间线整理

---

## 🎯 最终效果

**精简后的文档数量**:
- 根目录: 2个 (README.md, QUICK_START.md)
- docs/: ~15个 (分类清晰)
- archive/: ~50个 (历史记录)

**优点**:
1. ✅ 新用户能快速找到需要的文档
2. ✅ 文档职责清晰，不重复
3. ✅ 保留历史记录但不干扰主流程
4. ✅ 便于维护和更新

---

## 💡 下一步

1. **立即**: 创建 `archive/` 并移动过时文档
2. **今天**: 合并快速开始文档
3. **本周**: 重组 docs/ 结构
4. **持续**: 更新链接，确保文档间引用正确

需要帮助执行这些操作吗？
