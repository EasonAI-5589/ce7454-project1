# CE7454 Project 1 Technical Report

## 报告概述

本技术报告按照CVPR格式编写,详细阐述了基于MicroSegFormer的人脸解析方法。

## 文件结构

```
report/
├── main.tex              # 主文档
├── main.bib             # 参考文献
├── main.pdf             # 生成的PDF报告
├── cvpr.sty             # CVPR样式文件
├── sec/
│   ├── 0_abstract.tex   # 摘要
│   ├── 1_intro.tex      # 引言
│   ├── 2_method.tex     # 方法
│   ├── 3_experiments.tex # 实验
│   └── 4_conclusion.tex  # 结论
```

## 报告内容

### 1. Abstract (摘要)
- 问题定义和研究目标
- MicroSegFormer架构概述
- 主要贡献和结果

### 2. Introduction (引言) 
- 人脸解析任务背景
- 研究挑战(参数限制、细粒度分割、类别不平衡)
- 本文方法概述

### 3. Method (方法)
- **架构设计**: 层次化Transformer编码器 + MLP解码器
- **高效注意力机制**: 空间降维自注意力(SR ratios: [8,4,2,1])
- **损失函数**: CE Loss + 0.5×Dice Loss
- **优化策略**: AdamW + Cosine Annealing + Gradient Clipping
- **数据增强**: 几何变换(翻转、旋转、缩放) + 光度变换(颜色抖动)

### 4. Experiments (实验)
- **模型参数优化**: 1.72M参数(94.6%利用率)
- **损失函数消融**: CE+Dice组合最优
- **数据增强分析**: 几何+光度增强效果最佳
- **学习率调度**: Cosine annealing优于step decay
- **类别性能分析**: 大区域(背景、皮肤、头发)准确率高,小物体(耳环、项链)具有挑战性

### 5. Conclusion (结论)
- 主要贡献总结
- 实验洞察
- 局限性和未来工作方向

## 评分标准对应

根据评分标准,报告涵盖:

1. **Predictive Accuracy (30%)**: 
   - 模型架构设计(Section 2.1-2.3)
   - 实验结果分析(Section 3)

2. **Optimization and Regularization (20%)**:
   - 优化策略(Section 2.4)
   - 数据增强(Section 2.5)
   - 消融实验(Section 3.2-3.3)

3. **Experimental Analysis (30%)**:
   - 架构搜索(Table 1)
   - 损失函数对比(Table 2)
   - 数据增强分析(Table 3)
   - 学习率调度比较(Section 3.3)

4. **Clarity of Report (20%)**:
   - 清晰的结构组织
   - 详细的方法描述
   - 数学公式支撑
   - 实验数据表格

## 编译报告

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

生成的PDF: `main.pdf`

## 关键信息

- **模型**: MicroSegFormer
- **参数量**: 1,721,939 (94.6% of 1,821,085 limit)
- **数据集**: CelebAMask-HQ mini (1000 train + 100 val)
- **性能**: F-Score ~0.81 (validation)
- **优化器**: AdamW (lr=1.5e-3, wd=1e-4)
- **训练**: 150 epochs, batch size 32, A100 GPU

## 注意事项

1. 报告采用CVPR 2025模板格式
2. 页数限制4页(不含参考文献)
3. 字体要求: Times 10pt
4. 包含完整的方法描述和实验分析
5. 所有数值和表格基于实际实验结果

## 提交准备

报告已准备好提交,包含:
- ✅ 完整的技术描述
- ✅ 清晰的方法说明
- ✅ 详细的实验分析
- ✅ 适当的参考文献
- ✅ 符合CVPR格式要求
