# 优化计划 - 冲刺75% F-Score

## 当前状态
- Val F-Score: 64.83%
- 目标: 75%
- 差距: 10.17%

## 已完成优化 ✅
1. Ultra数据增强 (Flip, Rotation, Scale, Color Jitter, Noise)
2. 正则化增强 (Weight Decay 5x)
3. 混合精度训练 (AMP)
4. Early Stopping调整

## 进一步优化策略

### 策略1: 模型架构优化 (推荐)
- [ ] 添加Dropout层 (0.1-0.2)
- [ ] 在decoder中添加更多skip connections
- [ ] 调整feature fusion策略

### 策略2: 训练策略优化
- [ ] 增加训练轮数: 150 -> 200
- [ ] 调整学习率策略
- [ ] 使用Label Smoothing
- [ ] 尝试Focal Loss (处理类别不平衡)

### 策略3: 数据优化
- [ ] 增加CutMix/Mixup
- [ ] 更激进的数据增强
- [ ] Multi-scale training

### 策略4: 测试时增强 (TTA)
- [ ] 多尺度测试
- [ ] 水平翻转集成
- [ ] 这个可以在推理时使用，不影响训练

## 优先级排序
1. **立即**: 添加Dropout + 增加训练轮数
2. **次要**: Label Smoothing + Focal Loss
3. **最后**: TTA (测试时)

## 预期提升
- Dropout: +2-3%
- 更多训练轮数: +1-2%
- Label Smoothing: +1-2%
- TTA: +2-3%

**总计预期**: 64.83% + 6-10% = **70-75%**
