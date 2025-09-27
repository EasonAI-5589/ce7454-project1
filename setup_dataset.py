"""
Setup CE7454 dataset for training
Split 1000 train images into train/val sets as required
"""
import os
import shutil
import random
from pathlib import Path

def create_train_val_split(data_root, val_ratio=0.2):
    """
    Split the 1000 training images into train/val sets
    Following CodaBench guideline: "Divide the 1,000 images in the 'train' folder
    into a training set and a validation set"
    """

    # Original train folder
    orig_train_img = os.path.join(data_root, 'train', 'images')
    orig_train_mask = os.path.join(data_root, 'train', 'masks')

    # New directory structure
    new_train_img = os.path.join(data_root, 'train_split', 'images')
    new_train_mask = os.path.join(data_root, 'train_split', 'masks')
    new_val_img = os.path.join(data_root, 'val', 'images')
    new_val_mask = os.path.join(data_root, 'val', 'masks')

    # Create directories
    os.makedirs(new_train_img, exist_ok=True)
    os.makedirs(new_train_mask, exist_ok=True)
    os.makedirs(new_val_img, exist_ok=True)
    os.makedirs(new_val_mask, exist_ok=True)

    # Get all training images
    all_images = [f for f in os.listdir(orig_train_img) if f.endswith('.jpg')]
    print(f"Found {len(all_images)} training images")

    # Random split
    random.seed(42)  # For reproducibility
    val_count = int(len(all_images) * val_ratio)
    val_images = random.sample(all_images, val_count)
    train_images = [img for img in all_images if img not in val_images]

    print(f"Split: {len(train_images)} train, {len(val_images)} val")

    # Copy training images
    for img_file in train_images:
        mask_file = img_file.replace('.jpg', '.png')

        # Copy image
        src_img = os.path.join(orig_train_img, img_file)
        dst_img = os.path.join(new_train_img, img_file)
        shutil.copy2(src_img, dst_img)

        # Copy mask
        src_mask = os.path.join(orig_train_mask, mask_file)
        dst_mask = os.path.join(new_train_mask, mask_file)
        if os.path.exists(src_mask):
            shutil.copy2(src_mask, dst_mask)

    # Copy validation images
    for img_file in val_images:
        mask_file = img_file.replace('.jpg', '.png')

        # Copy image
        src_img = os.path.join(orig_train_img, img_file)
        dst_img = os.path.join(new_val_img, img_file)
        shutil.copy2(src_img, dst_img)

        # Copy mask
        src_mask = os.path.join(orig_train_mask, mask_file)
        dst_mask = os.path.join(new_val_mask, mask_file)
        if os.path.exists(src_mask):
            shutil.copy2(src_mask, dst_mask)

    print("Dataset split completed!")
    return len(train_images), len(val_images)

def verify_dataset_structure(data_root):
    """Verify the final dataset structure"""
    print("\n" + "="*50)
    print("DATASET VERIFICATION")
    print("="*50)

    # Check directories
    directories = [
        'train/images', 'train/masks',
        'train_split/images', 'train_split/masks',
        'val/images', 'val/masks',
        'test/images'
    ]

    for dir_path in directories:
        full_path = os.path.join(data_root, dir_path)
        if os.path.exists(full_path):
            count = len([f for f in os.listdir(full_path)
                        if f.endswith(('.jpg', '.png'))])
            print(f"✅ {dir_path:<20} -> {count:4d} files")
        else:
            print(f"❌ {dir_path:<20} -> NOT FOUND")

    # Check sample files
    print("\nSample filename verification:")
    train_imgs = os.path.join(data_root, 'train_split', 'images')
    train_masks = os.path.join(data_root, 'train_split', 'masks')

    if os.path.exists(train_imgs) and os.path.exists(train_masks):
        sample_img = os.listdir(train_imgs)[0]
        sample_mask = sample_img.replace('.jpg', '.png')

        img_exists = os.path.exists(os.path.join(train_imgs, sample_img))
        mask_exists = os.path.exists(os.path.join(train_masks, sample_mask))

        print(f"Sample image: {sample_img} -> {'✅' if img_exists else '❌'}")
        print(f"Sample mask:  {sample_mask} -> {'✅' if mask_exists else '❌'}")

def check_image_properties(data_root):
    """Check image and mask properties"""
    from PIL import Image
    import numpy as np

    print("\n" + "="*50)
    print("IMAGE PROPERTIES CHECK")
    print("="*50)

    # Check a sample image
    train_img_dir = os.path.join(data_root, 'train_split', 'images')
    train_mask_dir = os.path.join(data_root, 'train_split', 'masks')

    if os.path.exists(train_img_dir):
        sample_img_file = os.listdir(train_img_dir)[0]
        sample_mask_file = sample_img_file.replace('.jpg', '.png')

        # Load sample image
        img_path = os.path.join(train_img_dir, sample_img_file)
        mask_path = os.path.join(train_mask_dir, sample_mask_file)

        img = Image.open(img_path)
        mask = Image.open(mask_path)
        mask_array = np.array(mask)

        print(f"Image size: {img.size}")
        print(f"Image mode: {img.mode}")
        print(f"Mask size: {mask.size}")
        print(f"Mask mode: {mask.mode}")
        print(f"Mask unique values: {sorted(np.unique(mask_array))}")
        print(f"Mask max value: {mask_array.max()}")

if __name__ == "__main__":
    data_root = "data"

    print("Setting up CE7454 dataset for training...")

    # Create train/val split
    train_count, val_count = create_train_val_split(data_root, val_ratio=0.2)

    # Verify structure
    verify_dataset_structure(data_root)

    # Check properties
    try:
        check_image_properties(data_root)
    except Exception as e:
        print(f"Could not check image properties: {e}")

    print("\n" + "="*50)
    print("SETUP COMPLETE!")
    print("="*50)
    print(f"Ready to train with {train_count} train and {val_count} val images")
    print("Run: python src/train.py")