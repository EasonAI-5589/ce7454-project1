"""
GPU-based Advanced Data Augmentation
所有增强都在GPU上进行 (使用Kornia)

包括:
1. MixUp (GPU版本)
2. CutMix (GPU版本)
3. 强力几何+颜色增强 (Kornia)
"""
import torch
import torch.nn as nn
import kornia.augmentation as K
from kornia.augmentation import AugmentationSequential
import numpy as np


class GPUMixUp(nn.Module):
    """
    GPU版本的MixUp

    位置: 训练循环中，在数据移到GPU后应用
    作用: 混合batch内的样本，提高泛化能力
    """
    def __init__(self, alpha=0.2, prob=0.5):
        """
        Args:
            alpha: Beta分布参数 (0.2推荐用于分割)
            prob: 应用MixUp的概率
        """
        super().__init__()
        self.alpha = alpha
        self.prob = prob

    def forward(self, images, masks):
        """
        Args:
            images: (B, C, H, W) GPU tensor
            masks: (B, H, W) GPU long tensor
        Returns:
            mixed_images, mixed_masks
        """
        if self.alpha <= 0 or torch.rand(1).item() > self.prob:
            return images, masks

        batch_size = images.size(0)

        # 采样lambda
        lam = np.random.beta(self.alpha, self.alpha)

        # 随机排列
        index = torch.randperm(batch_size, device=images.device)

        # 混合图像
        mixed_images = lam * images + (1 - lam) * images[index]

        # 混合mask (硬标签)
        if lam > 0.5:
            mixed_masks = masks
        else:
            mixed_masks = masks[index]

        return mixed_images, mixed_masks


class GPUCutMix(nn.Module):
    """
    GPU版本的CutMix

    位置: 训练循环中，在数据移到GPU后应用
    作用: 裁剪粘贴，强化小目标学习
    """
    def __init__(self, alpha=1.0, prob=0.5):
        super().__init__()
        self.alpha = alpha
        self.prob = prob

    def forward(self, images, masks):
        """
        Args:
            images: (B, C, H, W) GPU tensor
            masks: (B, H, W) GPU long tensor
        """
        if torch.rand(1).item() > self.prob:
            return images, masks

        batch_size = images.size(0)
        _, _, H, W = images.shape

        # 采样lambda
        lam = np.random.beta(self.alpha, self.alpha)

        # 随机排列
        index = torch.randperm(batch_size, device=images.device)

        # 生成裁剪框
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        x1 = np.clip(cx - cut_w // 2, 0, W)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        # 执行CutMix
        mixed_images = images.clone()
        mixed_masks = masks.clone()

        mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        mixed_masks[:, y1:y2, x1:x2] = masks[index, y1:y2, x1:x2]

        return mixed_images, mixed_masks


class UltraStrongGPUAugmentation(nn.Module):
    """
    超强GPU增强 - 结合Kornia的高级增强

    对比现有的StrongGPUAugmentation:
    - 添加更激进的几何变换
    - 添加弹性形变(Elastic transform)的近似
    - 更强的颜色变换

    全部在GPU上运行!
    """
    def __init__(self):
        super().__init__()

        self.aug = AugmentationSequential(
            # 1. 几何增强 (更激进)
            K.RandomHorizontalFlip(p=0.5, same_on_batch=False),

            K.RandomRotation(
                degrees=25,  # 比原来的20度更大
                p=0.5,
                same_on_batch=False
            ),

            K.RandomAffine(
                degrees=0,
                scale=(0.8, 1.2),  # 更大范围
                translate=(0.1, 0.1),
                shear=10,  # 添加剪切变换
                p=0.5,
                same_on_batch=False
            ),

            # 2. 颜色增强 (更激进)
            K.ColorJitter(
                brightness=0.4,  # 更强
                contrast=0.4,
                saturation=0.2,
                hue=0.1,  # 添加色调变化
                p=0.7,
                same_on_batch=False
            ),

            # 3. 模糊和噪声
            K.RandomGaussianBlur(
                kernel_size=(3, 3),
                sigma=(0.1, 2.0),
                p=0.3,
                same_on_batch=False
            ),

            K.RandomGaussianNoise(
                mean=0.,
                std=0.05,
                p=0.2,
                same_on_batch=False
            ),

            # 4. Perspective变换 (模拟角度变化)
            K.RandomPerspective(
                distortion_scale=0.2,
                p=0.3,
                same_on_batch=False
            ),

            data_keys=["input", "mask"],
        )

    def forward(self, images, masks):
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)

        masks_float = masks.float()
        aug_images, aug_masks = self.aug(images, masks_float)
        aug_masks = aug_masks.long().squeeze(1)

        return aug_images, aug_masks


class CombinedAdvancedAugmentation(nn.Module):
    """
    组合增强模块 - 结合MixUp/CutMix和几何/颜色增强

    使用方式:
    ```python
    aug = CombinedAdvancedAugmentation(use_mixup=True, use_cutmix=True)

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        images, masks = aug(images, masks)  # 全部在GPU上!
        ...
    ```
    """
    def __init__(self,
                 use_mixup=True,
                 use_cutmix=True,
                 mixup_alpha=0.2,
                 cutmix_alpha=1.0,
                 mixup_prob=0.3,
                 cutmix_prob=0.3):
        super().__init__()

        # 基础Kornia增强
        self.kornia_aug = UltraStrongGPUAugmentation()

        # MixUp
        self.use_mixup = use_mixup
        if use_mixup:
            self.mixup = GPUMixUp(alpha=mixup_alpha, prob=mixup_prob)

        # CutMix
        self.use_cutmix = use_cutmix
        if use_cutmix:
            self.cutmix = GPUCutMix(alpha=cutmix_alpha, prob=cutmix_prob)

    def forward(self, images, masks):
        """
        应用顺序:
        1. Kornia几何+颜色增强
        2. MixUp或CutMix (随机选一个)
        """
        # Step 1: Kornia增强
        images, masks = self.kornia_aug(images, masks)

        # Step 2: MixUp或CutMix (随机)
        if self.use_mixup and self.use_cutmix:
            if torch.rand(1).item() > 0.5:
                images, masks = self.mixup(images, masks)
            else:
                images, masks = self.cutmix(images, masks)
        elif self.use_mixup:
            images, masks = self.mixup(images, masks)
        elif self.use_cutmix:
            images, masks = self.cutmix(images, masks)

        return images, masks


def test_gpu_advanced_augmentation():
    """测试GPU高级增强"""
    print("=" * 70)
    print("GPU高级数据增强测试")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n测试设备: {device}")

    if device.type == 'cpu':
        print("⚠️  警告: GPU不可用，在CPU上测试 (实际训练应在GPU上)")

    # 创建测试数据
    B, C, H, W = 8, 3, 512, 512
    images = torch.randn(B, C, H, W, device=device)
    masks = torch.randint(0, 19, (B, H, W), device=device)

    print(f"\n输入:")
    print(f"  Images: {tuple(images.shape)}")
    print(f"  Masks: {tuple(masks.shape)}")

    # 测试各个模块
    print("\n" + "-" * 70)
    print("1. GPUMixUp测试")
    mixup = GPUMixUp(alpha=0.2, prob=1.0).to(device)
    aug_img, aug_mask = mixup(images, masks)
    print(f"  输出: Images {tuple(aug_img.shape)}, Masks {tuple(aug_mask.shape)}")
    print(f"  ✓ GPUMixUp工作正常")

    print("\n" + "-" * 70)
    print("2. GPUCutMix测试")
    cutmix = GPUCutMix(alpha=1.0, prob=1.0).to(device)
    aug_img, aug_mask = cutmix(images, masks)
    print(f"  输出: Images {tuple(aug_img.shape)}, Masks {tuple(aug_mask.shape)}")
    print(f"  ✓ GPUCutMix工作正常")

    print("\n" + "-" * 70)
    print("3. UltraStrongGPUAugmentation测试")
    strong_aug = UltraStrongGPUAugmentation().to(device)
    aug_img, aug_mask = strong_aug(images, masks)
    print(f"  输出: Images {tuple(aug_img.shape)}, Masks {tuple(aug_mask.shape)}")
    print(f"  ✓ UltraStrongGPUAugmentation工作正常")

    print("\n" + "-" * 70)
    print("4. CombinedAdvancedAugmentation测试")
    combined_aug = CombinedAdvancedAugmentation(
        use_mixup=True,
        use_cutmix=True,
        mixup_prob=0.5,
        cutmix_prob=0.5
    ).to(device)

    # 性能测试
    import time
    n_iters = 50
    start = time.time()
    for _ in range(n_iters):
        aug_img, aug_mask = combined_aug(images, masks)
    end = time.time()

    print(f"  输出: Images {tuple(aug_img.shape)}, Masks {tuple(aug_mask.shape)}")
    print(f"  性能: {(end-start)/n_iters*1000:.2f} ms/batch (batch_size={B})")
    print(f"  ✓ CombinedAdvancedAugmentation工作正常")

    print("\n" + "=" * 70)
    print("所有GPU增强模块测试通过! ✓")
    print("=" * 70)

    print("\n使用示例:")
    print("""
    # 在trainer中集成:
    aug = CombinedAdvancedAugmentation(
        use_mixup=True,
        use_cutmix=True,
        mixup_prob=0.3,
        cutmix_prob=0.3
    ).to(device)

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        # 应用GPU增强 (全部在GPU上!)
        images, masks = aug(images, masks)

        # 正常训练
        outputs = model(images)
        loss = criterion(outputs, masks)
        ...
    """)


if __name__ == "__main__":
    test_gpu_advanced_augmentation()
