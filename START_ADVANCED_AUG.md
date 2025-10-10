# 🚀 高级数据增强训练 - 快速启动

## 📦 需要上传到服务器的文件

```
src/gpu_augmentation_torch.py           # 新增 - GPU增强模块
main.py                                  # 已修改 - 集成增强
configs/lmsa_advanced_aug.yaml          # 新增 - 训练配置
run_advanced_aug_training.sh            # 新增 - 启动脚本
```

---

## ⚡ 启动训练（3步）

### 1. SSH连接服务器
```bash
ssh your_username@server_address
cd /path/to/ce7454-project1
```

### 2. 启动训练
```bash
bash run_advanced_aug_training.sh
```

### 3. 查看实时日志
```bash
tail -f logs/lmsa_advanced_aug_*.log
```

---

## 📊 监控训练进度

### 实时日志
```bash
tail -f logs/lmsa_advanced_aug_*.log
```

### GPU使用情况
```bash
watch -n 1 nvidia-smi
```

### 检查进程
```bash
ps aux | grep "main.py"
```

---

## ⏸️ 控制训练

### 停止训练
```bash
pkill -f "main.py.*lmsa_advanced_aug"
```

### 恢复训练
```bash
python main.py --config configs/lmsa_advanced_aug.yaml \
    --resume checkpoints/microsegformer_XXXXXX/last_model.pth
```

---

## ✅ 预期结果

| 指标 | 当前 | 目标 | 提升 |
|------|------|------|------|
| Val F-Score | 0.7041 | 0.71-0.72 | +1% |
| Test F-Score | 0.72 | 0.73-0.75 | +3-4% |
| Val-Test Gap | 1.6% | <1% | 缩小 |

**训练时间:** 约6-10小时 (200 epochs)

---

## 🎯 技术亮点

✅ **MixUp** (30%概率) - 混合图像，防止过拟合
✅ **CutMix** (30%概率) - 裁剪粘贴，强化小目标
✅ **全GPU实现** - 无CPU瓶颈
✅ **参数不变** - 1.75M (在限制内)

---

## 📖 详细文档

完整说明: [ADVANCED_AUG_README.md](ADVANCED_AUG_README.md)

---

祝训练顺利！🎉
