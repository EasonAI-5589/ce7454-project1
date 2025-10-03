# MicroSegFormer Training Guide

完整的MicroSegFormer训练流程指南。

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 检查数据集
ls data/train/images/  # 应该有1000张训练图像
ls data/train/masks/   # 应该有对应的mask文件
```

### 2. 开始训练

```bash
# 最简单的方式
./quick_start.sh train

# 或使用Python直接运行
python main.py --config configs/main.yaml
```

## 📋 训练配置

### 默认配置 (configs/main.yaml)

```yaml
model:
  name: microsegformer
  num_classes: 19

data:
  root: data
  batch_size: 8
  num_workers: 4
  val_split: 0.1

training:
  epochs: 150
  optimizer: AdamW
  learning_rate: 1e-3
  weight_decay: 1e-4
  scheduler: CosineAnnealingLR
  warmup_epochs: 5
  early_stopping_patience: 20

loss:
  ce_weight: 1.0
  dice_weight: 0.5

augmentation:
  horizontal_flip: 0.5
  rotation: 15
  color_jitter:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.1
  scale_range: [0.9, 1.1]
```

### 调整配置

直接编辑 `configs/main.yaml` 文件，然后重新训练。

**常用调整**:
```yaml
# 增加batch size (如果GPU内存充足)
batch_size: 16

# 调整学习率
learning_rate: 5e-4

# 延长训练
epochs: 200

# 调整early stopping
early_stopping_patience: 30
```

## 🎯 训练命令

### 从头训练

```bash
./quick_start.sh train
```

或使用指定设备：
```bash
python main.py --config configs/main.yaml --device cuda:0
```

### 恢复训练

```bash
# 从checkpoint恢复
./quick_start.sh resume checkpoints/best_model.pth

# 或指定完整路径
python main.py \
  --config configs/main.yaml \
  --resume checkpoints/microsegformer_20241004_120000/best_model.pth
```

### 训练输出

训练过程会生成以下文件：

```
checkpoints/microsegformer_YYYYMMDD_HHMMSS/
├── best_model.pth          # 最佳模型（验证集F-Score最高）
├── last_model.pth          # 最后一个epoch的模型
├── config.yaml             # 训练配置备份
└── training_log.txt        # 训练日志
```

checkpoint中包含：
- `model_state_dict`: 模型权重
- `optimizer_state_dict`: 优化器状态
- `scheduler_state_dict`: 学习率调度器状态
- `epoch`: 当前epoch
- `best_f_score`: 最佳F-Score
- `config`: 配置信息

## 🔍 监控训练

### 实时查看训练日志

```bash
tail -f checkpoints/microsegformer_*/training_log.txt
```

### 关键指标

训练时会输出：
- **Train Loss**: 训练损失（CE Loss + Dice Loss）
- **Val F-Score**: 验证集F-Score（主要评估指标）
- **Learning Rate**: 当前学习率
- **Epoch Time**: 每个epoch耗时

示例输出：
```
Epoch [10/150] - Train Loss: 0.342, Val F-Score: 0.7856, LR: 9.5e-4, Time: 45.2s
Epoch [20/150] - Train Loss: 0.298, Val F-Score: 0.8123, LR: 8.8e-4, Time: 44.8s
```

## 📊 测试模型

### 评估验证集

```bash
./quick_start.sh test checkpoints/best_model.pth

# 或
python test.py --checkpoint checkpoints/best_model.pth
```

输出示例：
```
Loaded model from: checkpoints/best_model.pth
Epoch: 85
Best F-Score: 0.8234

Validation set size: 100
Testing: 100%|████████████| 13/13 [00:15<00:00]

Validation F-Score: 0.8234
```

## 💡 训练技巧

### 1. 提升性能的策略

**数据增强**:
```yaml
# 增强更激进的数据增强
augmentation:
  horizontal_flip: 0.5
  rotation: 20              # 增加旋转角度
  color_jitter:
    brightness: 0.3         # 增加亮度变化
    contrast: 0.3
    saturation: 0.2
  scale_range: [0.8, 1.2]   # 增大缩放范围
```

**学习率调优**:
```yaml
# 降低初始学习率
learning_rate: 5e-4

# 或使用warmup
warmup_epochs: 10
```

**Loss权重调整**:
```yaml
# 增加Dice Loss权重
loss:
  ce_weight: 1.0
  dice_weight: 1.0         # 从0.5增加到1.0
```

### 2. 加速训练

**增加batch size**:
```yaml
# 如果GPU内存允许
batch_size: 16  # 或32
```

**减少验证频率**:
- 修改 `src/trainer.py` 中的验证频率
- 每2-3个epoch验证一次

**使用混合精度** (如果支持):
```yaml
training:
  use_amp: true
```

### 3. 处理过拟合

如果验证F-Score不再提升：

- 增加数据增强强度
- 增加weight decay: `1e-3`
- 减少模型容量（但要注意参数限制）
- 使用early stopping（已启用）

### 4. 处理欠拟合

如果训练和验证F-Score都很低：

- 增加训练epochs
- 提高学习率: `2e-3`
- 检查数据质量
- 减少weight decay: `1e-5`

## 🎓 训练流程说明

### 完整训练流程

1. **初始化**
   - 加载配置
   - 创建模型（验证参数量）
   - 准备数据集（90% train / 10% val）
   - 初始化优化器和调度器

2. **训练循环**
   ```
   for epoch in 1..150:
       - 训练一个epoch
       - 计算训练损失
       - 在验证集上评估F-Score
       - 更新学习率（CosineAnnealing）
       - 保存best model（如果F-Score提升）
       - 检查early stopping
   ```

3. **Early Stopping**
   - 如果验证F-Score连续20个epoch没有提升
   - 自动停止训练
   - 避免过拟合和浪费时间

### 学习率调度

使用CosineAnnealingLR：
- 开始: 1e-3
- 逐渐降低到接近0
- 前5个epoch有warmup

### 损失函数

组合损失:
```python
total_loss = ce_loss * 1.0 + dice_loss * 0.5
```

- **CrossEntropy**: 逐像素分类
- **Dice Loss**: 关注区域重叠

## 🐛 常见问题

### CUDA Out of Memory
```bash
# 减小batch size
batch_size: 4
```

### 训练太慢
```bash
# 减少workers或检查数据加载
num_workers: 2
```

### 验证F-Score不稳定
- 增加验证集大小: `val_split: 0.15`
- 或使用固定的验证集

### 模型不收敛
- 检查数据归一化
- 降低学习率
- 检查label是否正确

## 📚 下一步

训练完成后：
1. 使用best_model.pth在测试集上预测
2. 提交到Codabench
3. 根据结果调整配置
4. 重新训练
