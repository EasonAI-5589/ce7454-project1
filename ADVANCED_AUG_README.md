# 高级数据增强训练指南

## 🎯 改进方案概述

**问题诊断:**
- 当前模型: Val 0.7041 → Test 0.72 (过拟合，泛化能力不足)
- 生成的mask边界粗糙

**解决方案:**
- ✅ 使用MixUp + CutMix高级数据增强
- ✅ 全部在GPU上运行（无CPU瓶颈）
- ✅ 参数完全不变（1.75M，在1.82M限制内）

**预期效果:**
- Val 0.71-0.72 → Test 0.73-0.75
- 缩小Val-Test gap
- 提高泛化能力

---

## 📂 新增文件

1. **src/gpu_augmentation_torch.py** - 纯PyTorch实现的GPU增强
   - `TorchGPUMixUp`: MixUp增强
   - `TorchGPUCutMix`: CutMix增强
   - `CombinedAdvancedAugmentation`: 组合增强模块

2. **configs/lmsa_advanced_aug.yaml** - 高级增强训练配置
   - MixUp: prob=0.3, alpha=0.2
   - CutMix: prob=0.3, alpha=1.0
   - 其他参数保持v3最佳配置

3. **run_advanced_aug_training.sh** - 一键启动脚本

---

## 🚀 在服务器上启动训练

### 方法1: 使用启动脚本（推荐）

```bash
# 1. 上传代码到服务器
# 确保以下文件都上传了：
#   - src/gpu_augmentation_torch.py (新增)
#   - main.py (已修改)
#   - configs/lmsa_advanced_aug.yaml (新增)
#   - run_advanced_aug_training.sh (新增)

# 2. 给脚本执行权限
chmod +x run_advanced_aug_training.sh

# 3. 启动训练
bash run_advanced_aug_training.sh

# 4. 脚本会自动：
#   - 检查GPU
#   - 创建日志文件
#   - 后台启动训练
#   - 显示监控命令
```

### 方法2: 手动启动

```bash
# 创建日志目录
mkdir -p logs

# 后台启动训练
nohup python main.py --config configs/lmsa_advanced_aug.yaml > logs/advanced_aug.log 2>&1 &

# 记录进程ID
echo $! > logs/training_pid.txt

# 查看实时日志
tail -f logs/advanced_aug.log
```

---

## 📊 监控训练进度

### 查看实时日志
```bash
tail -f logs/lmsa_advanced_aug_*.log
```

### 查看最近100行
```bash
tail -100 logs/lmsa_advanced_aug_*.log
```

### 检查进程是否运行
```bash
# 查看进程
ps aux | grep "main.py"

# 或使用保存的PID
ps -p $(cat logs/training_pid.txt)
```

### 监控GPU使用
```bash
# 实时监控
watch -n 1 nvidia-smi

# 或
nvidia-smi -l 1
```

### 停止训练
```bash
# 使用保存的PID
kill $(cat logs/training_pid.txt)

# 或直接kill进程
pkill -f "main.py.*lmsa_advanced_aug"
```

---

## 📈 训练效果分析

### 检查checkpoint

训练完成后，模型会保存在 `checkpoints/microsegformer_<timestamp>/`

```bash
# 查看所有checkpoints
ls -lh checkpoints/

# 查看最新的checkpoint
ls -lht checkpoints/ | head -5

# 查看训练历史
python -c "
import torch
ckpt = torch.load('checkpoints/microsegformer_XXX/best_model.pth')
print('Best F-Score:', ckpt['best_f_score'])
print('Epoch:', ckpt['epoch'])
"
```

### 提取训练曲线

```bash
# 从日志提取F-Score
grep "F-Score" logs/advanced_aug.log | tail -20

# 提取Val F-Score
grep "Val F-Score" logs/advanced_aug.log
```

---

## 🔍 对比实验

| 实验 | Val F-Score | Test F-Score | Gap | 备注 |
|------|-------------|--------------|-----|------|
| v1 (baseline) | 0.6819 | 0.72 | +5.6% | 泛化好但性能低 |
| v3 (当前最佳) | 0.7041 | 0.72 | +1.6% | 过拟合 |
| **advanced_aug (新)** | **0.71-0.72** | **0.73-0.75** | **<1%** | **目标** |

---

## 🛠️ 技术细节

### MixUp原理
```python
# 混合两张图像
new_img = λ * img1 + (1-λ) * img2
new_mask = mask1 (if λ>0.5) else mask2

# 效果：
# - 生成新的训练样本
# - 防止模型记住特定样本
# - 提高泛化能力
```

### CutMix原理
```python
# 从img2裁剪矩形区域，粘贴到img1
# mask也同步替换

# 效果：
# - 强化局部特征学习
# - 对小目标(眼睛、嘴巴)特别有效
# - 增加边界训练样本
```

### GPU增强流程
```
训练循环:
  for images, masks in dataloader:
      images, masks = images.to(gpu)

      # GPU增强 (全部在GPU上!)
      images, masks = advanced_aug(images, masks)

      # 正常训练
      outputs = model(images)
      loss = criterion(outputs, masks)
      ...
```

---

## ❓ 常见问题

### Q1: 训练很慢怎么办？
A: 检查以下几点：
- GPU是否被正确使用：`nvidia-smi`
- batch_size是否合适（默认32）
- num_workers是否合理（默认4）

### Q2: Out of Memory错误
A: 减小batch_size：
```yaml
# 在configs/lmsa_advanced_aug.yaml中
data:
  batch_size: 16  # 从32改为16
```

### Q3: 如何调整增强强度？
A: 修改配置文件：
```yaml
training:
  mixup_prob: 0.5    # 增加到50%
  cutmix_prob: 0.5   # 增加到50%
```

### Q4: 训练中断如何恢复？
A: 使用--resume参数：
```bash
python main.py --config configs/lmsa_advanced_aug.yaml \
    --resume checkpoints/microsegformer_XXX/last_model.pth
```

---

## 📝 提交到Codabench

训练完成后：

```bash
# 1. 生成测试集预测
python inference.py \
    --model-path checkpoints/microsegformer_XXX/best_model.pth \
    --output-dir submissions/submission_v5_advanced_aug

# 2. 打包提交文件
cd submissions/submission_v5_advanced_aug
zip -r ../submission_v5_advanced_aug.zip masks/

# 3. 上传到Codabench
# 文件: submission_v5_advanced_aug.zip
```

---

## 📧 联系

如有问题，请检查：
1. 日志文件: `logs/lmsa_advanced_aug_*.log`
2. 配置文件: `configs/lmsa_advanced_aug.yaml`
3. 代码: `src/gpu_augmentation_torch.py`

预计训练时间: 200 epochs × ~2-3分钟/epoch = 6-10小时

祝训练顺利！🎉
