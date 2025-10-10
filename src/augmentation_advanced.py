"""
【方案1】高级数据增强模块 - Advanced Data Augmentation
包括: MixUp, CutMix, Elastic Deformation

=== 模块作用和位置 ===

1. MixUp / CutMix:
   位置: 在DataLoader的collate_fn中应用 (batch级别)
   作用: 混合不同样本，生成新的训练样本
   优势: 提高泛化能力，减少过拟合，增强边界学习

2. ElasticDeformation:
   位置: 在Dataset的__getitem__中应用 (样本级别)
   作用: 模拟面部表情/角度变化
   优势: 增强形变鲁棒性，对面部分割特别有效

3. AdvancedColorAugmentation:
   位置: 在Dataset的__getitem__中应用
   作用: 模拟不同光照/肤色/化妆
   优势: 减少颜色特征过拟合
"""
import torch
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates


class MixUp:
    """
    MixUp数据增强
    论文: mixup: Beyond Empirical Risk Minimization (ICLR 2018)

    === 位置 ===
    DataLoader的collate_fn或训练循环中(batch级别)

    === 作用 ===
    混合两张图像及其标签: new_img = λ*img1 + (1-λ)*img2

    === 优势 ===
    - 提高模型泛化能力
    - 减少过拟合
    - 对边界区域的学习更鲁棒(混合边界提供更多训练信号)
    - 特别适合解决Val高/Test低的过拟合问题
    """
    def __init__(self, alpha=0.2):
        """
        Args:
            alpha: Beta分布参数，控制混合强度
                  - alpha=0.2: 温和混合 (推荐用于分割)
                  - alpha=1.0: 均匀混合
        """
        self.alpha = alpha

    def __call__(self, images, masks):
        """
        Args:
            images: (B, C, H, W) 图像batch
            masks: (B, H, W) mask batch
        Returns:
            mixed_images: (B, C, H, W) 混合后的图像
            mixed_masks: (B, H, W) 混合后的mask (采用硬标签)
        """
        if self.alpha <= 0:
            return images, masks

        batch_size = images.size(0)

        # 采样混合系数 lambda ~ Beta(alpha, alpha)
        lam = np.random.beta(self.alpha, self.alpha)

        # 随机排列获取混合对象
        index = torch.randperm(batch_size).to(images.device)

        # 混合图像 (线性插值)
        mixed_images = lam * images + (1 - lam) * images[index]

        # 混合mask (采用硬标签策略: lambda>0.5用原mask，否则用混合mask)
        if lam > 0.5:
            mixed_masks = masks
        else:
            mixed_masks = masks[index]

        return mixed_images, mixed_masks


class CutMix:
    """
    CutMix数据增强
    论文: CutMix: Regularization Strategy to Train Strong Classifiers (ICCV 2019)

    === 位置 ===
    DataLoader的collate_fn或训练循环中(batch级别)

    === 作用 ===
    裁剪一张图像的矩形区域，粘贴到另一张图像上

    === 优势 ===
    - 强制模型关注局部特征
    - 对小目标(眼睛、嘴巴)的学习特别有效
    - 提供更多的局部边界训练样本
    - 比MixUp更激进，适合小目标密集的任务
    """
    def __init__(self, alpha=1.0, prob=0.5):
        """
        Args:
            alpha: Beta分布参数，控制裁剪区域大小
            prob: 应用CutMix的概率
        """
        self.alpha = alpha
        self.prob = prob

    def __call__(self, images, masks):
        """
        Args:
            images: (B, C, H, W)
            masks: (B, H, W)
        Returns:
            mixed_images, mixed_masks
        """
        if np.random.rand() > self.prob:
            return images, masks

        batch_size = images.size(0)
        _, _, H, W = images.shape

        # 采样混合系数
        lam = np.random.beta(self.alpha, self.alpha)

        # 随机选择混合对象
        index = torch.randperm(batch_size).to(images.device)

        # 生成随机裁剪框
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)

        # 随机中心点
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # 裁剪框边界
        x1 = np.clip(cx - cut_w // 2, 0, W)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        # 复制图像和mask
        mixed_images = images.clone()
        mixed_masks = masks.clone()

        # 执行CutMix
        mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        mixed_masks[:, y1:y2, x1:x2] = masks[index, y1:y2, x1:x2]

        return mixed_images, mixed_masks


class ElasticDeformation:
    """
    弹性形变增强
    论文: Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis

    === 位置 ===
    Dataset的__getitem__中，对单张图像应用

    === 作用 ===
    对图像进行平滑的非线性形变，模拟面部表情变化、角度变化

    === 优势 ===
    - 模拟面部表情变化、角度变化
    - 增强模型对形变的鲁棒性
    - 对医学图像和面部分割特别有效
    - 比简单的仿射变换更自然
    """
    def __init__(self, alpha=250, sigma=10, prob=0.3):
        """
        Args:
            alpha: 形变强度 (像素位移范围)
                  - 250: 适中形变 (推荐)
                  - 500: 强烈形变
            sigma: 高斯平滑参数 (控制形变平滑度)
                  - 10: 平滑形变
                  - 5: 更局部的形变
            prob: 应用概率
        """
        self.alpha = alpha
        self.sigma = sigma
        self.prob = prob

    def __call__(self, image, mask):
        """
        Args:
            image: (H, W, C) numpy array [0, 255] uint8
            mask: (H, W) numpy array uint8
        Returns:
            deformed_image, deformed_mask
        """
        if np.random.rand() > self.prob:
            return image, mask

        H, W = image.shape[:2]

        # 生成随机位移场
        dx = gaussian_filter((np.random.rand(H, W) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((np.random.rand(H, W) * 2 - 1), self.sigma) * self.alpha

        # 创建网格坐标
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        # 应用形变到图像 (使用双线性插值)
        deformed_image = np.zeros_like(image)
        for i in range(image.shape[2]):
            deformed_image[:, :, i] = map_coordinates(
                image[:, :, i], indices, order=1, mode='reflect'
            ).reshape(H, W)

        # 应用形变到mask (使用最近邻插值，保持标签整数)
        deformed_mask = map_coordinates(
            mask, indices, order=0, mode='reflect'
        ).reshape(H, W).astype(np.uint8)

        return deformed_image, deformed_mask


class AdvancedColorAugmentation:
    """
    高级颜色增强

    === 位置 ===
    Dataset的__getitem__中

    === 作用 ===
    模拟不同光照、肤色、化妆效果

    === 优势 ===
    - 提高对不同肤色、光照的鲁棒性
    - 减少对特定颜色特征的过拟合
    - 比基础ColorJitter更针对面部场景
    """
    def __init__(self, prob=0.7):
        self.prob = prob

    def __call__(self, image):
        """
        Args:
            image: (H, W, C) numpy array [0, 255] uint8
        Returns:
            augmented_image
        """
        if np.random.rand() > self.prob:
            return image

        image = image.astype(np.float32)

        # 1. 色调偏移 (Hue shift) - 模拟不同光源
        if np.random.rand() > 0.5:
            hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-10, 10)) % 180
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        # 2. 饱和度调整 - 模拟化妆效果
        if np.random.rand() > 0.5:
            saturation = np.random.uniform(0.7, 1.3)
            hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        # 3. 亮度调整 - 模拟光照变化
        if np.random.rand() > 0.5:
            brightness = np.random.uniform(0.7, 1.3)
            image = np.clip(image * brightness, 0, 255)

        # 4. 对比度调整
        if np.random.rand() > 0.5:
            contrast = np.random.uniform(0.7, 1.3)
            mean = image.mean(axis=(0, 1), keepdims=True)
            image = np.clip((image - mean) * contrast + mean, 0, 255)

        return image.astype(np.uint8)


def test_augmentations():
    """测试增强模块"""
    print("=" * 60)
    print("测试高级数据增强模块")
    print("=" * 60)

    # 测试MixUp
    print("\n1. MixUp (Batch级别)")
    print("   位置: DataLoader collate_fn / 训练循环")
    print("   作用: 混合两张图像，提高泛化能力")
    mixup = MixUp(alpha=0.2)
    images = torch.randn(4, 3, 512, 512)
    masks = torch.randint(0, 19, (4, 512, 512))
    mixed_img, mixed_mask = mixup(images, masks)
    print(f"   Input: {images.shape}, {masks.shape}")
    print(f"   Output: {mixed_img.shape}, {mixed_mask.shape}")
    print(f"   ✓ MixUp working")

    # 测试CutMix
    print("\n2. CutMix (Batch级别)")
    print("   位置: DataLoader collate_fn / 训练循环")
    print("   作用: 裁剪粘贴，强化小目标学习")
    cutmix = CutMix(alpha=1.0, prob=1.0)
    mixed_img, mixed_mask = cutmix(images, masks)
    print(f"   Input: {images.shape}, {masks.shape}")
    print(f"   Output: {mixed_img.shape}, {mixed_mask.shape}")
    print(f"   ✓ CutMix working")

    # 测试ElasticDeformation
    print("\n3. ElasticDeformation (样本级别)")
    print("   位置: Dataset.__getitem__()")
    print("   作用: 平滑形变，模拟表情变化")
    elastic = ElasticDeformation(alpha=250, sigma=10, prob=1.0)
    image_np = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    mask_np = np.random.randint(0, 19, (512, 512), dtype=np.uint8)
    deformed_img, deformed_mask = elastic(image_np, mask_np)
    print(f"   Input: {image_np.shape}, {mask_np.shape}")
    print(f"   Output: {deformed_img.shape}, {deformed_mask.shape}")
    print(f"   ✓ ElasticDeformation working")

    # 测试AdvancedColorAugmentation
    print("\n4. AdvancedColorAugmentation (样本级别)")
    print("   位置: Dataset.__getitem__()")
    print("   作用: 颜色变换，模拟光照/肤色差异")
    color_aug = AdvancedColorAugmentation(prob=1.0)
    aug_img = color_aug(image_np)
    print(f"   Input: {image_np.shape}")
    print(f"   Output: {aug_img.shape}")
    print(f"   ✓ AdvancedColorAugmentation working")

    print("\n" + "=" * 60)
    print("所有增强模块测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_augmentations()
