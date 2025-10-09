"""
Ultra-Strong Data Augmentation for Face Parsing
Includes: Flip, Rotation, Scale, Color Jitter, Gaussian Blur
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random

class FaceParsingDataset(Dataset):
    """Face parsing dataset with ultra-strong augmentation"""

    def __init__(self, data_root, split='train', image_size=512, augment=False):
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.augment = augment

        # Build file lists
        if split == 'train':
            self.image_dir = os.path.join(data_root, 'train', 'images')
            self.mask_dir = os.path.join(data_root, 'train', 'masks')

            self.image_files = [f for f in os.listdir(self.image_dir)
                              if f.endswith('.jpg')]
            self.image_files.sort()

        elif split == 'test':
            self.image_dir = os.path.join(data_root, 'test', 'images')
            self.mask_dir = None

            self.image_files = [f for f in os.listdir(self.image_dir)
                              if f.endswith('.jpg')]
            self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def _apply_augmentation(self, image, mask):
        """ULTRA-STRONG augmentation pipeline"""
        # 1. Random Horizontal Flip (p=0.5)
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        # 2. Random Rotation (-20° to +20°)
        if random.random() > 0.4:
            angle = random.uniform(-20, 20)
            image = self._rotate(image, angle)
            mask = self._rotate(mask, angle, is_mask=True)

        # 3. Random Scale (0.8 to 1.2)
        if random.random() > 0.4:
            scale = random.uniform(0.8, 1.2)
            image = self._scale_crop(image, scale)
            mask = self._scale_crop(mask, scale, is_mask=True)

        # 4. Color Jitter (brightness, contrast)
        if random.random() > 0.5:
            # Brightness
            brightness = random.uniform(0.7, 1.3)
            image = np.clip(image.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

            # Contrast
            if random.random() > 0.5:
                contrast = random.uniform(0.8, 1.2)
                mean = image.mean()
                image = np.clip((image - mean) * contrast + mean, 0, 255).astype(np.uint8)

        # 5. Random Gaussian Noise
        if random.random() > 0.7:
            noise = np.random.normal(0, random.uniform(3, 8), image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return image, mask

    def _rotate(self, img, angle, is_mask=False):
        """Rotate image using PIL"""
        if is_mask:
            img_pil = Image.fromarray(img.astype(np.uint8))
            rotated = img_pil.rotate(angle, resample=Image.NEAREST, fillcolor=0)
        else:
            img_pil = Image.fromarray(img)
            rotated = img_pil.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
        return np.array(rotated)

    def _scale_crop(self, img, scale, is_mask=False):
        """Scale and center crop to original size"""
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        if is_mask:
            img_pil = Image.fromarray(img.astype(np.uint8))
            resized = img_pil.resize((new_w, new_h), resample=Image.NEAREST)
        else:
            img_pil = Image.fromarray(img)
            resized = img_pil.resize((new_w, new_h), resample=Image.BILINEAR)

        resized_np = np.array(resized)

        # Center crop or pad to original size
        if scale > 1:  # Crop
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            return resized_np[start_h:start_h+h, start_w:start_w+w]
        else:  # Pad
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            if len(img.shape) == 3:
                padded = np.zeros((h, w, img.shape[2]), dtype=img.dtype)
                padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized_np
            else:
                padded = np.zeros((h, w), dtype=img.dtype)
                padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized_np
            return padded

    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Resize image
        image = image.resize((self.image_size, self.image_size))

        # Load mask if available
        if self.mask_dir is not None:
            mask_name = img_name.replace('.jpg', '.png')
            mask_path = os.path.join(self.mask_dir, mask_name)

            if os.path.exists(mask_path):
                mask = Image.open(mask_path)
                mask = mask.resize((self.image_size, self.image_size),
                                 resample=Image.NEAREST)
                mask = np.array(mask)
            else:
                mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        else:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        # Convert to arrays
        image = np.array(image)

        # ULTRA Data Augmentation
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)

        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Convert to PyTorch tensors
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        mask = torch.from_numpy(mask).long()

        return image, mask

def get_dataloader(data_root, split='train', batch_size=8, num_workers=2, augment=False):
    """Create dataloader"""
    dataset = FaceParsingDataset(data_root, split=split, augment=augment)

    shuffle = (split == 'train')
    pin_memory = torch.cuda.is_available()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return dataloader


def create_train_val_loaders(data_root, batch_size=8, num_workers=2, val_split=0.1, seed=42):
    """Create train and validation dataloaders"""
    # Create separate datasets - train with augmentation, val without
    train_dataset_full = FaceParsingDataset(data_root, split='train', augment=True)
    val_dataset_full = FaceParsingDataset(data_root, split='train', augment=False)

    # Calculate split sizes
    total_size = len(train_dataset_full)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    print(f"Splitting dataset: {total_size} total -> {train_size} train, {val_size} val")

    # Get indices for split
    import numpy as np
    np.random.seed(seed)
    indices = np.random.permutation(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create subset datasets
    from torch.utils.data import Subset
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    pin_memory = torch.cuda.is_available()

    # Optimize num_workers to avoid CPU bottleneck
    # Rule of thumb: 4-8 workers is usually optimal
    optimal_workers = min(num_workers, 8)
    if num_workers > 8:
        print(f"  ⚠️  num_workers={num_workers} too high, using {optimal_workers} for better performance")
        num_workers = optimal_workers

    # Create dataloaders with optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch 2 batches per worker
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
        drop_last=True  # Drop incomplete last batch for consistent training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False  # Keep all validation data
    )

    return train_loader, val_loader

def test_dataset(data_root='data'):
    """Test dataset loading"""
    print("Testing dataset loading...")

    # Test train split
    try:
        train_dataset = FaceParsingDataset(data_root, split='train')
        print(f"✓ Train dataset: {len(train_dataset)} samples")

        # Test loading one sample
        image, mask = train_dataset[0]
        print(f"✓ Image shape: {image.shape}, dtype: {image.dtype}")
        print(f"✓ Mask shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"✓ Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"✓ Mask range: [{mask.min()}, {mask.max()}]")

    except Exception as e:
        print(f"✗ Train dataset error: {e}")
        return False

    # Test test split
    try:
        test_dataset = FaceParsingDataset(data_root, split='test')
        print(f"✓ Test dataset: {len(test_dataset)} samples")

    except Exception as e:
        print(f"✗ Test dataset error: {e}")
        return False

    # Test dataloader
    try:
        train_loader = get_dataloader(data_root, split='train', batch_size=2)
        for images, masks in train_loader:
            print(f"✓ Batch - Images: {images.shape}, Masks: {masks.shape}")
            break

    except Exception as e:
        print(f"✗ DataLoader error: {e}")
        return False

    print("✓ Dataset test passed!")
    return True

if __name__ == "__main__":
    test_dataset()
