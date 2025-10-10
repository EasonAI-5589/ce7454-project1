# GPU占用率不高问题分析与优化方案
**日期**: 2025-10-10

---

## 🔍 问题诊断

通过代码审查，我发现了**5个主要的GPU性能瓶颈**：

---

### 1️⃣ **CPU数据增强** ⚠️ 最严重瓶颈

**问题位置**: [src/dataset.py:41-77](src/dataset.py#L41-L77)

**问题**:
```python
def _apply_augmentation(self, image, mask):
    """ULTRA-STRONG augmentation pipeline"""
    # ❌ 全部在CPU上执行的NumPy操作
    if random.random() > 0.5:
        image = np.fliplr(image).copy()  # CPU
        mask = np.fliplr(mask).copy()    # CPU

    if random.random() > 0.4:
        angle = random.uniform(-20, 20)
        image = self._rotate(image, angle)      # PIL + CPU
        mask = self._rotate(mask, angle, is_mask=True)

    if random.random() > 0.4:
        scale = random.uniform(0.8, 1.2)
        image = self._scale_crop(image, scale)  # PIL resize
        mask = self._scale_crop(mask, scale, is_mask=True)

    # Color jitter, Gaussian noise... 都在CPU上
```

**影响**:
- 🐢 数据增强在CPU上执行
- 🐢 PIL/NumPy操作非常慢 (5-10ms/图像)
- 🐢 GPU等待CPU完成数据准备
- 🐢 **batch_size=32时，每个batch需要160-320ms仅用于数据增强**

**GPU利用率**: **<30%** (大部分时间在等待CPU)

---

### 2️⃣ **num_workers不足** ⚠️

**问题位置**: [main.py:96-102](main.py#L96-L102)

**当前配置**:
```python
train_loader, val_loader = create_train_val_loaders(
    data_root=config['data']['root'],
    batch_size=config['data']['batch_size'],  # 32
    num_workers=config['data']['num_workers'], # ❌ 只有4个
    val_split=config['data']['val_split']
)
```

**问题**:
- 4个worker并行加载数据不够快
- CPU预处理速度 < GPU训练速度
- GPU经常空闲等待下一个batch

**推荐**: 8-12个workers (取决于CPU核心数)

---

### 3️⃣ **没有pin_memory** ⚠️

**问题**:
```python
# DataLoader没有启用pin_memory
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
    # ❌ 缺少: pin_memory=True
)
```

**影响**:
- CPU内存 → GPU内存的传输速度慢
- 每个batch传输时间增加5-10ms

---

### 4️⃣ **数据传输同步阻塞** ⚠️

**问题位置**: [src/trainer.py:97-99](src/trainer.py#L97-L99)

```python
for batch_idx, (images, masks) in enumerate(self.train_loader):
    images = images.to(self.device)  # ❌ 同步阻塞
    masks = masks.to(self.device)    # ❌ 同步阻塞
```

**问题**:
- `.to(device)` 默认是同步操作
- 没有使用 `non_blocking=True`
- GPU等待数据传输完成才开始计算

---

### 5️⃣ **没有persistent_workers** ⚠️ (轻微)

**问题**:
- 每个epoch结束后worker进程会销毁并重新创建
- 造成额外的启动开销

---

## 📊 性能影响估算

| 瓶颈 | GPU空闲时间/Batch | 影响程度 |
|------|------------------|---------|
| CPU数据增强 | 160-320ms | ⭐⭐⭐⭐⭐ 最严重 |
| num_workers不足 | 50-100ms | ⭐⭐⭐⭐ |
| 没有pin_memory | 5-10ms | ⭐⭐ |
| 同步数据传输 | 5-10ms | ⭐⭐ |
| 没有persistent_workers | 每epoch 100-200ms | ⭐ |
| **总计** | **220-440ms/batch** | **GPU利用率 <40%** |

**计算**:
- 假设GPU前向+反向传播需要 100ms
- 数据准备需要 220-440ms
- GPU利用率 = 100 / (100 + 300) = **25-30%** ✅ 符合你的观察

---

## ✅ 优化方案

### 🚀 方案1: 将数据增强移到GPU (推荐) ⭐⭐⭐⭐⭐

**效果**: GPU利用率 30% → **85-95%**

#### 实现方式A: 使用 Kornia (最推荐)

**安装**:
```bash
pip install kornia
```

**新建文件**: `src/gpu_augmentation.py`

```python
"""GPU-accelerated data augmentation using Kornia"""
import torch
import torch.nn as nn
import kornia as K
import kornia.augmentation as KA


class GPUAugmentation(nn.Module):
    """GPU-based augmentation pipeline using Kornia"""

    def __init__(self):
        super().__init__()

        # All operations run on GPU
        self.aug = nn.Sequential(
            # Geometric transforms
            KA.RandomHorizontalFlip(p=0.5),
            KA.RandomRotation(degrees=20, p=0.6),
            KA.RandomAffine(
                degrees=0,
                scale=(0.8, 1.2),
                p=0.6
            ),

            # Color transforms
            KA.ColorJitter(
                brightness=0.3,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),

            # Blur and noise
            KA.RandomGaussianBlur(
                kernel_size=(3, 3),
                sigma=(0.1, 2.0),
                p=0.3
            ),
            KA.RandomGaussianNoise(
                mean=0.,
                std=0.03,
                p=0.3
            ),
        )

    def forward(self, images, masks):
        """
        Args:
            images: [B, 3, H, W] float32 [0, 1]
            masks: [B, H, W] long [0, 18]
        Returns:
            augmented images and masks
        """
        # Stack images and masks for synchronized augmentation
        # Masks need to be converted to float for kornia
        masks_float = masks.unsqueeze(1).float()  # [B, 1, H, W]

        # Concatenate along channel dimension
        combined = torch.cat([images, masks_float], dim=1)  # [B, 4, H, W]

        # Apply augmentation (geometric transforms affect all channels)
        augmented = self.aug(combined)

        # Split back
        aug_images = augmented[:, :3]  # [B, 3, H, W]
        aug_masks = augmented[:, 3:4].squeeze(1).long()  # [B, H, W]

        # Ensure mask values stay in valid range [0, 18]
        aug_masks = torch.clamp(aug_masks, 0, 18)

        return aug_images, aug_masks


class GPUAugmentationAdvanced(nn.Module):
    """Advanced GPU augmentation with MixUp/CutMix"""

    def __init__(
        self,
        use_mixup=True,
        use_cutmix=True,
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
        mixup_prob=0.3,
        cutmix_prob=0.3
    ):
        super().__init__()

        # Basic augmentation
        self.basic_aug = GPUAugmentation()

        # Advanced augmentation
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob

        # Kornia's mixup and cutmix
        if use_mixup:
            self.mixup = KA.RandomMixUpV2(
                lambda_val=(mixup_alpha, mixup_alpha),
                p=mixup_prob
            )

        if use_cutmix:
            self.cutmix = KA.RandomCutMixV2(
                cut_size=(0.2, 0.8),
                p=cutmix_prob
            )

    def forward(self, images, masks):
        # Apply basic augmentation
        images, masks = self.basic_aug(images, masks)

        # Apply mixup (50% chance)
        if self.use_mixup and torch.rand(1).item() < self.mixup_prob:
            images = self.mixup(images)

        # Apply cutmix (50% chance)
        if self.use_cutmix and torch.rand(1).item() < self.cutmix_prob:
            images = self.cutmix(images)

        return images, masks
```

#### 修改 `src/dataset.py`

```python
class FaceParsingDataset(Dataset):
    """Face parsing dataset - NO CPU AUGMENTATION"""

    def __init__(self, data_root, split='train', image_size=512, augment=False):
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        # ❌ 移除所有 CPU 增强代码
        self.augment = False  # 强制关闭CPU增强

        # ... (其他代码保持不变)

    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size))

        # Load mask
        if self.mask_dir is not None:
            mask_name = img_name.replace('.jpg', '.png')
            mask_path = os.path.join(self.mask_dir, mask_name)
            mask = Image.open(mask_path)
            mask = mask.resize((self.image_size, self.image_size),
                             resample=Image.NEAREST)
            mask = np.array(mask)
        else:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        # Convert to arrays
        image = np.array(image)

        # ❌ 删除: if self.augment: image, mask = self._apply_augmentation(...)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1)  # [3, H, W]
        mask = torch.from_numpy(mask).long()  # [H, W]

        return image, mask
```

#### 修改 `src/trainer.py`

```python
def train_epoch(self):
    """Train for one epoch with GPU augmentation"""
    self.model.train()

    # ✅ 初始化GPU增强 (在 __init__ 中)
    # self.gpu_aug = GPUAugmentation().to(self.device)

    losses = AverageMeter()
    f_scores = AverageMeter()

    for batch_idx, (images, masks) in enumerate(self.train_loader):
        # ✅ 使用 non_blocking 加速传输
        images = images.to(self.device, non_blocking=True)
        masks = masks.to(self.device, non_blocking=True)

        # ✅ GPU增强 (超快!)
        with torch.no_grad():
            images, masks = self.gpu_aug(images, masks)

        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, masks)

        # Backward
        loss.backward()
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
        self.optimizer.step()

        # ... (metrics calculation)
```

#### 修改 `main.py`

```python
from src.gpu_augmentation import GPUAugmentation, GPUAugmentationAdvanced

def main():
    # ... (load config, setup model)

    # Create data loaders (✅ 移除CPU增强)
    train_loader, val_loader = create_train_val_loaders(
        data_root=config['data']['root'],
        batch_size=config['data']['batch_size'],
        num_workers=8,  # ✅ 增加workers
        val_split=config['data']['val_split']
    )

    # ✅ Setup GPU augmentation
    gpu_aug = GPUAugmentation().to(device)
    print("✓ GPU augmentation enabled (Kornia)")

    # Setup trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        gpu_augmentation=gpu_aug  # ✅ 传入GPU增强
    )

    # Start training
    trainer.fit(epochs=config['training']['epochs'])
```

---

### 🚀 方案2: 优化DataLoader配置 ⭐⭐⭐⭐

#### 修改 `src/dataset.py` 中的 `create_train_val_loaders`

```python
def create_train_val_loaders(
    data_root='data',
    batch_size=32,
    num_workers=8,  # ✅ 增加到8
    val_split=0.1,
    image_size=512
):
    """Create train and validation data loaders with optimizations"""

    # Create datasets (no CPU augmentation)
    train_dataset = FaceParsingDataset(
        data_root=data_root,
        split='train',
        image_size=image_size,
        augment=False  # ✅ 关闭CPU增强
    )

    # Split into train/val
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    # ✅ 优化的DataLoader配置
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # 8-12个workers
        pin_memory=True,          # ✅ 启用pin_memory
        persistent_workers=True,  # ✅ 保持workers alive
        prefetch_factor=2,        # ✅ 预加载2个batch
        drop_last=True            # ✅ 避免最后不完整batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers // 2,  # 验证可以少一些
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    return train_loader, val_loader
```

---

### 🚀 方案3: 启用混合精度训练 ⭐⭐⭐

**当前状态**: 代码已经支持，但需要确保配置启用

#### 检查配置文件 `configs/lmsa.yaml`

```yaml
training:
  use_amp: true  # ✅ 确保启用
  # ...
```

**效果**:
- 训练速度提升 2-3x
- GPU内存使用减少 ~50%
- 可以增加 batch_size

---

## 📋 完整优化清单

### 立即执行 (今天)

- [ ] **1. 安装Kornia**
  ```bash
  pip install kornia
  ```

- [ ] **2. 创建GPU增强模块**
  - 新建 `src/gpu_augmentation.py`
  - 实现 `GPUAugmentation` 类

- [ ] **3. 修改 dataset.py**
  - 关闭CPU增强
  - 优化DataLoader配置 (pin_memory, persistent_workers)

- [ ] **4. 修改 trainer.py**
  - 添加GPU增强调用
  - 使用 `non_blocking=True`

- [ ] **5. 更新配置文件**
  ```yaml
  data:
    batch_size: 32  # 或更大(如果GPU内存允许)
    num_workers: 8  # 增加到8-12

  training:
    use_amp: true
    use_gpu_augmentation: true
  ```

- [ ] **6. 测试训练**
  ```bash
  python main.py --config configs/lmsa_gpu_optimized.yaml
  ```

---

## 🎯 预期效果

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| **GPU利用率** | 25-30% | 85-95% | **+3x** |
| **训练速度** | ~30s/epoch | ~10s/epoch | **+3x** |
| **每秒样本数** | ~33 samples/s | ~100 samples/s | **+3x** |
| **总训练时间** | 150 epoch = 75min | 150 epoch = 25min | **节省50分钟** |

---

## 🔧 故障排查

### 问题1: Kornia安装失败
```bash
# 使用清华镜像
pip install kornia -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题2: GPU内存不足
```yaml
# 降低batch_size
data:
  batch_size: 16  # 从32降到16
```

### 问题3: num_workers过多导致CPU瓶颈
```yaml
# 根据CPU核心数调整
data:
  num_workers: 4  # CPU 6核 → 4 workers
  num_workers: 8  # CPU 12核 → 8 workers
```

---

## 💡 额外优化 (可选)

### 1. 使用DALI (最极致性能)

NVIDIA DALI提供更快的数据加载，但配置复杂：
```bash
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda120
```

### 2. 编译PyTorch模型

```python
# 使用torch.compile (PyTorch 2.0+)
model = torch.compile(model, mode='max-autotune')
```

### 3. 使用更大的batch_size

GPU增强后，可以尝试：
```yaml
data:
  batch_size: 48  # 或64，取决于GPU内存
```

---

## 🎓 总结

**核心问题**: CPU数据增强成为瓶颈，GPU大部分时间在等待数据

**解决方案**:
1. ⭐⭐⭐⭐⭐ **将数据增强移到GPU** (Kornia)
2. ⭐⭐⭐⭐ **优化DataLoader** (pin_memory, more workers)
3. ⭐⭐⭐ **混合精度训练** (AMP)

**预期提升**: GPU利用率 30% → **90%**，训练速度 **+3倍**

---

需要我帮你实现这些优化吗？
