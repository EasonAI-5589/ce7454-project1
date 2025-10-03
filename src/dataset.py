"""
Simple dataset loader for face parsing
Only depends on PIL, numpy and PyTorch core
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random

class FaceParsingDataset(Dataset):
    """Simple face parsing dataset"""

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

        # Simple augmentation
        if self.augment and random.random() > 0.5:
            # Horizontal flip
            image = np.fliplr(image)
            mask = np.fliplr(mask)

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
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def create_train_val_loaders(data_root, batch_size=8, num_workers=2, val_split=0.1, seed=42):
    """
    Create train and validation dataloaders by splitting the training set

    Args:
        data_root: Root directory of dataset
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        val_split: Fraction of data to use for validation (default 0.1 = 10%)
        seed: Random seed for reproducible split

    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import random_split

    # Create full training dataset
    full_dataset = FaceParsingDataset(data_root, split='train', augment=True)

    # Calculate split sizes
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    print(f"Splitting dataset: {total_size} total -> {train_size} train, {val_size} val")

    # Split dataset
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
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