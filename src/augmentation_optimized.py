"""
GPU-friendly and CPU-optimized augmentation pipeline
使用更快的操作来减少CPU瓶颈
"""
import torch
import torchvision.transforms.functional as TF
import numpy as np
import random
from PIL import Image


class FastAugmentation:
    """
    优化的数据增强pipeline
    - 减少PIL/numpy转换
    - 使用更快的操作
    - 支持批量处理
    """
    
    def __init__(self, 
                 horizontal_flip_prob=0.5,
                 rotation_range=15,
                 scale_range=(0.9, 1.1),
                 brightness=0.2,
                 contrast=0.2):
        self.horizontal_flip_prob = horizontal_flip_prob
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.brightness = brightness
        self.contrast = contrast
    
    def __call__(self, image, mask):
        """
        Apply augmentation to image and mask
        
        Args:
            image: PIL Image or numpy array (H, W, C)
            mask: PIL Image or numpy array (H, W)
        
        Returns:
            image: augmented image (PIL Image)
            mask: augmented mask (PIL Image)
        """
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask.astype(np.uint8))
        
        # 1. Horizontal Flip (fast)
        if random.random() < self.horizontal_flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # 2. Random Rotation (occasional, expensive)
        if random.random() < 0.3:  # 降低概率减少计算
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            image = TF.rotate(image, angle, fill=0)
            mask = TF.rotate(mask, angle, fill=0)
        
        # 3. Random Scale (occasional)
        if random.random() < 0.3:  # 降低概率
            scale = random.uniform(*self.scale_range)
            w, h = image.size
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Resize
            image = TF.resize(image, (new_h, new_w))
            mask = TF.resize(mask, (new_h, new_w), interpolation=TF.InterpolationMode.NEAREST)
            
            # Center crop/pad back to original size
            image = TF.center_crop(image, (h, w)) if scale > 1 else TF.pad(image, 
                [(w - new_w) // 2, (h - new_h) // 2, (w - new_w + 1) // 2, (h - new_h + 1) // 2])
            mask = TF.center_crop(mask, (h, w)) if scale > 1 else TF.pad(mask,
                [(w - new_w) // 2, (h - new_h) // 2, (w - new_w + 1) // 2, (h - new_h + 1) // 2])
        
        # 4. Color Jitter (cheap, only on image)
        if random.random() < 0.5:
            # Brightness
            if self.brightness > 0:
                factor = random.uniform(1 - self.brightness, 1 + self.brightness)
                image = TF.adjust_brightness(image, factor)
            
            # Contrast
            if self.contrast > 0 and random.random() < 0.5:
                factor = random.uniform(1 - self.contrast, 1 + self.contrast)
                image = TF.adjust_contrast(image, factor)
        
        return image, mask


class MinimalAugmentation:
    """
    最小化增强 - 只做最必要的变换
    用于快速训练或调试
    """
    
    def __init__(self, horizontal_flip_prob=0.5):
        self.horizontal_flip_prob = horizontal_flip_prob
    
    def __call__(self, image, mask):
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask.astype(np.uint8))
        
        # Only horizontal flip (cheapest operation)
        if random.random() < self.horizontal_flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        return image, mask
