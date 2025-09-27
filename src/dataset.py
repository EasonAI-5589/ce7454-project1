"""
Dataset loader for CE7454 Face Parsing Project
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from config import Config

class CelebAMaskDataset(Dataset):
    """CelebAMask-HQ dataset for face parsing"""

    def __init__(self, data_root, split='train', transform=None, target_transform=None):
        """
        Args:
            data_root (str): Root directory of dataset
            split (str): 'train', 'val', or 'test'
            transform: Transform to apply to images
            target_transform: Transform to apply to masks
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # Expected directory structure
        self.image_dir = os.path.join(data_root, split, 'images')
        self.mask_dir = os.path.join(data_root, split, 'masks')

        # Get list of images
        if os.path.exists(self.image_dir):
            self.image_files = sorted([f for f in os.listdir(self.image_dir)
                                     if f.endswith(('.png', '.jpg', '.jpeg'))])
        else:
            print(f"Warning: Image directory {self.image_dir} does not exist")
            self.image_files = []

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Load mask
        mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')  # Grayscale
            mask = np.array(mask)
        else:
            # Create dummy mask if not available
            mask = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.uint8)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = torch.from_numpy(mask).long()

        return image, mask

def get_transforms(split='train'):
    """Get data transforms for different splits"""

    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=Config.HORIZONTAL_FLIP_PROB),
            transforms.RandomRotation(degrees=Config.ROTATION_DEGREES),
            transforms.ColorJitter(
                brightness=Config.COLOR_JITTER_BRIGHTNESS,
                contrast=Config.COLOR_JITTER_CONTRAST
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN, std=Config.STD)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN, std=Config.STD)
        ])

    return transform

def get_dataloader(data_root, split='train', batch_size=None):
    """Get DataLoader for specified split"""

    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    transform = get_transforms(split)
    dataset = CelebAMaskDataset(data_root, split=split, transform=transform)

    shuffle = (split == 'train')
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,  # Reduced for stability
        pin_memory=True
    )

    return dataloader

def verify_dataset(data_root):
    """Verify dataset structure and contents"""
    print("Verifying dataset structure...")

    splits = ['train', 'val', 'test']
    for split in splits:
        image_dir = os.path.join(data_root, split, 'images')
        mask_dir = os.path.join(data_root, split, 'masks')

        if os.path.exists(image_dir):
            num_images = len([f for f in os.listdir(image_dir)
                            if f.endswith(('.png', '.jpg', '.jpeg'))])
            print(f"{split} images: {num_images}")
        else:
            print(f"{split} image directory not found: {image_dir}")

        if os.path.exists(mask_dir):
            num_masks = len([f for f in os.listdir(mask_dir)
                           if f.endswith('.png')])
            print(f"{split} masks: {num_masks}")
        else:
            print(f"{split} mask directory not found: {mask_dir}")

if __name__ == "__main__":
    # Test dataset loading
    data_root = Config.DATA_ROOT
    verify_dataset(data_root)