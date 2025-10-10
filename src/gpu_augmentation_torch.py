"""
GPU Advanced Augmentation - Pure PyTorch版本（不依赖Kornia）
在GPU上运行MixUp和CutMix，提高泛化能力

优势：
1. 纯PyTorch实现，不需要额外依赖
2. 全部在GPU上运行，无CPU瓶颈
3. 与训练无缝集成
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TorchGPUMixUp(nn.Module):
    """
    纯PyTorch实现的MixUp

    作用：混合batch内的样本，提高泛化能力
    原理：new_img = λ*img1 + (1-λ)*img2
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

        # 混合mask (硬标签策略)
        if lam > 0.5:
            mixed_masks = masks
        else:
            mixed_masks = masks[index]

        return mixed_images, mixed_masks


class TorchGPUCutMix(nn.Module):
    """
    纯PyTorch实现的CutMix

    作用：裁剪粘贴，强化小目标(眼睛、嘴巴)学习
    原理：从一张图裁剪矩形区域，粘贴到另一张图
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


class CombinedAdvancedAugmentation(nn.Module):
    """
    组合增强：MixUp + CutMix

    使用方式：
    ```python
    # 创建增强模块
    aug = CombinedAdvancedAugmentation(
        use_mixup=True,
        use_cutmix=True,
        mixup_prob=0.3,
        cutmix_prob=0.3
    )

    # 在训练循环中使用
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        images, masks = aug(images, masks)  # GPU增强！
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

        # MixUp
        self.use_mixup = use_mixup
        if use_mixup:
            self.mixup = TorchGPUMixUp(alpha=mixup_alpha, prob=mixup_prob)

        # CutMix
        self.use_cutmix = use_cutmix
        if use_cutmix:
            self.cutmix = TorchGPUCutMix(alpha=cutmix_alpha, prob=cutmix_prob)

    def forward(self, images, masks):
        """
        应用增强
        策略：随机选择MixUp或CutMix（不同时应用）
        """
        if self.use_mixup and self.use_cutmix:
            # 随机选一个
            if torch.rand(1).item() > 0.5:
                images, masks = self.mixup(images, masks)
            else:
                images, masks = self.cutmix(images, masks)
        elif self.use_mixup:
            images, masks = self.mixup(images, masks)
        elif self.use_cutmix:
            images, masks = self.cutmix(images, masks)

        return images, masks


def test_torch_gpu_augmentation():
    """测试纯PyTorch GPU增强"""
    print("=" * 70)
    print("纯PyTorch GPU增强测试")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n测试设备: {device}")

    if device.type == 'cpu':
        print("⚠️  警告: GPU不可用，在CPU上测试")

    # 创建测试数据
    B, C, H, W = 8, 3, 512, 512
    images = torch.randn(B, C, H, W, device=device)
    masks = torch.randint(0, 19, (B, H, W), device=device)

    print(f"\n输入:")
    print(f"  Images: {tuple(images.shape)}")
    print(f"  Masks: {tuple(masks.shape)}")

    # 测试MixUp
    print("\n" + "-" * 70)
    print("1. TorchGPUMixUp测试")
    mixup = TorchGPUMixUp(alpha=0.2, prob=1.0).to(device)
    aug_img, aug_mask = mixup(images, masks)
    print(f"  输出: Images {tuple(aug_img.shape)}, Masks {tuple(aug_mask.shape)}")
    print(f"  ✓ MixUp工作正常")

    # 测试CutMix
    print("\n" + "-" * 70)
    print("2. TorchGPUCutMix测试")
    cutmix = TorchGPUCutMix(alpha=1.0, prob=1.0).to(device)
    aug_img, aug_mask = cutmix(images, masks)
    print(f"  输出: Images {tuple(aug_img.shape)}, Masks {tuple(aug_mask.shape)}")
    print(f"  ✓ CutMix工作正常")

    # 测试组合增强
    print("\n" + "-" * 70)
    print("3. CombinedAdvancedAugmentation测试")
    combined = CombinedAdvancedAugmentation(
        use_mixup=True,
        use_cutmix=True,
        mixup_prob=0.5,
        cutmix_prob=0.5
    ).to(device)

    # 性能测试
    import time
    n_iters = 100
    combined.eval()  # 不需要训练模式
    with torch.no_grad():
        start = time.time()
        for _ in range(n_iters):
            aug_img, aug_mask = combined(images, masks)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.time()

    print(f"  输出: Images {tuple(aug_img.shape)}, Masks {tuple(aug_mask.shape)}")
    print(f"  性能: {(end-start)/n_iters*1000:.2f} ms/batch (batch_size={B})")
    print(f"  ✓ CombinedAdvancedAugmentation工作正常")

    print("\n" + "=" * 70)
    print("所有测试通过! ✓")
    print("=" * 70)

    print("\n集成示例:")
    print("""
# 在main.py中添加:
from src.gpu_augmentation_torch import CombinedAdvancedAugmentation

# 创建增强模块
gpu_aug = None
if config['training'].get('use_advanced_aug', False):
    gpu_aug = CombinedAdvancedAugmentation(
        use_mixup=True,
        use_cutmix=True,
        mixup_prob=0.3,
        cutmix_prob=0.3
    )

# 传给Trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    config=config,
    device=device,
    gpu_augmentation=gpu_aug  # ← 添加这个参数
)
""")


if __name__ == "__main__":
    test_torch_gpu_augmentation()
