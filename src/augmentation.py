"""
Enhanced Data Augmentation for Face Parsing
Supports both basic PyTorch transforms and advanced augmentations
"""
import torch
import numpy as np
import random
from PIL import Image, ImageEnhance


class RandomHorizontalFlip:
    """Random horizontal flip for both image and mask"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        return image, mask


class RandomRotation:
    """Random rotation for both image and mask"""
    def __init__(self, degrees=15):
        self.degrees = degrees

    def __call__(self, image, mask):
        if random.random() < 0.5:
            angle = random.uniform(-self.degrees, self.degrees)

            # Convert to PIL
            image_pil = Image.fromarray(image.astype(np.uint8))
            mask_pil = Image.fromarray(mask.astype(np.uint8))

            # Rotate
            image_pil = image_pil.rotate(angle, resample=Image.BILINEAR)
            mask_pil = mask_pil.rotate(angle, resample=Image.NEAREST)

            # Back to numpy
            image = np.array(image_pil)
            mask = np.array(mask_pil)

        return image, mask


class ColorJitter:
    """Color jittering (brightness, contrast, saturation)"""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, image, mask):
        if random.random() < 0.5:
            image_pil = Image.fromarray(image.astype(np.uint8))

            # Random brightness
            if self.brightness > 0:
                factor = random.uniform(1 - self.brightness, 1 + self.brightness)
                image_pil = ImageEnhance.Brightness(image_pil).enhance(factor)

            # Random contrast
            if self.contrast > 0:
                factor = random.uniform(1 - self.contrast, 1 + self.contrast)
                image_pil = ImageEnhance.Contrast(image_pil).enhance(factor)

            # Random saturation
            if self.saturation > 0:
                factor = random.uniform(1 - self.saturation, 1 + self.saturation)
                image_pil = ImageEnhance.Color(image_pil).enhance(factor)

            image = np.array(image_pil)

        return image, mask


class RandomScale:
    """Random scaling"""
    def __init__(self, scale_range=(0.9, 1.1)):
        self.scale_range = scale_range

    def __call__(self, image, mask):
        if random.random() < 0.5:
            scale = random.uniform(*self.scale_range)
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)

            # Resize
            image_pil = Image.fromarray(image.astype(np.uint8))
            mask_pil = Image.fromarray(mask.astype(np.uint8))

            image_pil = image_pil.resize((new_w, new_h), Image.BILINEAR)
            mask_pil = mask_pil.resize((new_w, new_h), Image.NEAREST)

            # Crop or pad back to original size
            image = np.array(image_pil)
            mask = np.array(mask_pil)

            if scale > 1:
                # Crop center
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                image = image[start_h:start_h+h, start_w:start_w+w]
                mask = mask[start_h:start_h+h, start_w:start_w+w]
            else:
                # Pad
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                image = np.pad(image, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w), (0, 0)), mode='reflect')
                mask = np.pad(mask, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w)), mode='reflect')

        return image, mask


class Compose:
    """Compose multiple augmentations"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class Normalize:
    """Normalize image"""
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array(mean).reshape(1, 1, 3)
        self.std = np.array(std).reshape(1, 1, 3)

    def __call__(self, image):
        # Assume image is already in [0, 1] range
        image = (image - self.mean) / self.std
        return image


def get_train_augmentation(config=None):
    """Get training augmentation pipeline"""
    if config is None:
        config = {
            'horizontal_flip': 0.5,
            'rotation': 15,
            'color_jitter': {
                'brightness': 0.2,
                'contrast': 0.2,
                'saturation': 0.1
            },
            'scale_range': (0.9, 1.1)
        }

    transforms = []

    # Geometric transforms
    if config.get('horizontal_flip', 0) > 0:
        transforms.append(RandomHorizontalFlip(p=config['horizontal_flip']))

    if config.get('rotation', 0) > 0:
        transforms.append(RandomRotation(degrees=config['rotation']))

    if config.get('scale_range'):
        transforms.append(RandomScale(scale_range=config['scale_range']))

    # Color transforms
    if config.get('color_jitter'):
        cj = config['color_jitter']
        transforms.append(ColorJitter(
            brightness=cj.get('brightness', 0.2),
            contrast=cj.get('contrast', 0.2),
            saturation=cj.get('saturation', 0.1)
        ))

    return Compose(transforms)


def get_val_augmentation():
    """Get validation augmentation (no augmentation)"""
    return Compose([])  # No augmentation for validation


if __name__ == "__main__":
    # Test augmentations
    print("Testing augmentations...")

    # Create dummy data
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    mask = np.random.randint(0, 19, (512, 512), dtype=np.uint8)

    # Test transforms
    aug = get_train_augmentation()
    image_aug, mask_aug = aug(image, mask)

    print(f"Original image shape: {image.shape}, dtype: {image.dtype}")
    print(f"Augmented image shape: {image_aug.shape}, dtype: {image_aug.dtype}")
    print(f"Original mask shape: {mask.shape}, unique: {np.unique(mask)[:5]}")
    print(f"Augmented mask shape: {mask_aug.shape}, unique: {np.unique(mask_aug)[:5]}")
    print("✓ Augmentation test passed!")
