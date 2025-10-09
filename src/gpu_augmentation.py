"""
GPU-based data augmentation using Kornia
所有增强操作都在GPU上进行,消除CPU瓶颈
"""
import torch
import torch.nn as nn
import kornia.augmentation as K
from kornia.augmentation import AugmentationSequential


class GPUAugmentation(nn.Module):
    """
    GPU-based augmentation pipeline using Kornia
    
    优点:
    1. 所有操作在GPU上,无CPU瓶颈
    2. 批量处理,充分利用GPU并行
    3. 与训练无缝集成
    4. 支持mixed precision
    """
    
    def __init__(self, 
                 horizontal_flip_prob=0.5,
                 rotation_degrees=15,
                 brightness=0.2,
                 contrast=0.2,
                 saturation=0.1):
        super().__init__()
        
        # 创建augmentation sequence
        # 注意: same_on_batch=False 让每张图片有不同的增强
        self.aug = AugmentationSequential(
            # 1. Random Horizontal Flip (cheap)
            K.RandomHorizontalFlip(p=horizontal_flip_prob, same_on_batch=False),
            
            # 2. Random Rotation (moderate cost)
            K.RandomRotation(degrees=rotation_degrees, p=0.3, same_on_batch=False),
            
            # 3. Random Affine (scale, translate)
            K.RandomAffine(
                degrees=0,
                scale=(0.9, 1.1),
                translate=(0.05, 0.05),
                p=0.3,
                same_on_batch=False
            ),
            
            # 4. Color Jitter (cheap, GPU-friendly)
            K.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                p=0.5,
                same_on_batch=False
            ),
            
            data_keys=["input", "mask"],  # 同时处理image和mask
        )
    
    def forward(self, images, masks):
        """
        Apply augmentation on GPU
        
        Args:
            images: (B, C, H, W) float tensor on GPU
            masks: (B, H, W) or (B, 1, H, W) long tensor on GPU
        
        Returns:
            aug_images: (B, C, H, W) augmented images
            aug_masks: (B, H, W) augmented masks
        """
        # Ensure masks have correct shape (B, 1, H, W)
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        
        # Convert mask to float for augmentation
        masks_float = masks.float()
        
        # Apply augmentation
        aug_images, aug_masks = self.aug(images, masks_float)
        
        # Convert mask back to long
        aug_masks = aug_masks.long().squeeze(1)  # (B, 1, H, W) -> (B, H, W)
        
        return aug_images, aug_masks


class MinimalGPUAugmentation(nn.Module):
    """
    最小化GPU增强 - 只做horizontal flip
    用于快速训练或当GPU内存紧张时
    """
    
    def __init__(self, horizontal_flip_prob=0.5):
        super().__init__()
        self.flip = K.RandomHorizontalFlip(p=horizontal_flip_prob, same_on_batch=False)
    
    def forward(self, images, masks):
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        
        masks_float = masks.float()
        aug_images, aug_masks = self.flip(images, masks_float)
        aug_masks = aug_masks.long().squeeze(1)
        
        return aug_images, aug_masks


class StrongGPUAugmentation(nn.Module):
    """
    强力GPU增强 - 用于提升泛化能力
    包含更多变换但仍然在GPU上高效运行
    """
    
    def __init__(self):
        super().__init__()
        
        self.aug = AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5, same_on_batch=False),
            K.RandomRotation(degrees=20, p=0.4, same_on_batch=False),
            K.RandomAffine(
                degrees=0,
                scale=(0.85, 1.15),
                translate=(0.1, 0.1),
                p=0.4,
                same_on_batch=False
            ),
            K.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.15,
                p=0.5,
                same_on_batch=False
            ),
            # 可选: 添加更多增强
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.2, same_on_batch=False),
            
            data_keys=["input", "mask"],
        )
    
    def forward(self, images, masks):
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        
        masks_float = masks.float()
        aug_images, aug_masks = self.aug(images, masks_float)
        aug_masks = aug_masks.long().squeeze(1)
        
        return aug_images, aug_masks


def test_gpu_augmentation():
    """Test GPU augmentation pipeline"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Create dummy data
    batch_size = 8
    images = torch.randn(batch_size, 3, 512, 512, device=device)
    masks = torch.randint(0, 19, (batch_size, 512, 512), device=device)
    
    # Test augmentation
    aug = GPUAugmentation().to(device)
    
    import time
    start = time.time()
    for _ in range(100):
        aug_images, aug_masks = aug(images, masks)
    end = time.time()
    
    print(f"GPU Augmentation: {(end-start)/100*1000:.2f} ms/batch")
    print(f"Output shapes: images {aug_images.shape}, masks {aug_masks.shape}")
    print("✅ GPU augmentation working!")


if __name__ == '__main__':
    test_gpu_augmentation()
