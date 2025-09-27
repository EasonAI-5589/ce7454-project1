"""
Basic test without external dependencies
"""
import os
from PIL import Image
import numpy as np

def test_dataset_structure():
    """Test dataset structure"""
    print("="*50)
    print("Testing Dataset Structure")
    print("="*50)

    data_root = "data"

    # Check directories
    train_img_dir = os.path.join(data_root, 'train', 'images')
    train_mask_dir = os.path.join(data_root, 'train', 'masks')
    test_img_dir = os.path.join(data_root, 'test', 'images')

    # Count files
    train_imgs = len([f for f in os.listdir(train_img_dir) if f.endswith('.jpg')])
    train_masks = len([f for f in os.listdir(train_mask_dir) if f.endswith('.png')])
    test_imgs = len([f for f in os.listdir(test_img_dir) if f.endswith('.jpg')])

    print(f"✓ Training images: {train_imgs}")
    print(f"✓ Training masks: {train_masks}")
    print(f"✓ Test images: {test_imgs}")

    # Check file matching
    sample_img = os.listdir(train_img_dir)[0]
    sample_mask = sample_img.replace('.jpg', '.png')

    img_exists = os.path.exists(os.path.join(train_img_dir, sample_img))
    mask_exists = os.path.exists(os.path.join(train_mask_dir, sample_mask))

    print(f"✓ Sample image exists: {img_exists}")
    print(f"✓ Sample mask exists: {mask_exists}")

    return train_imgs == 1000 and train_masks == 1000 and test_imgs == 100

def test_image_properties():
    """Test image and mask properties"""
    print("\n" + "="*50)
    print("Testing Image Properties")
    print("="*50)

    data_root = "data"
    train_img_dir = os.path.join(data_root, 'train', 'images')
    train_mask_dir = os.path.join(data_root, 'train', 'masks')

    # Load sample image and mask
    sample_img_file = os.listdir(train_img_dir)[0]
    sample_mask_file = sample_img_file.replace('.jpg', '.png')

    img_path = os.path.join(train_img_dir, sample_img_file)
    mask_path = os.path.join(train_mask_dir, sample_mask_file)

    try:
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        mask_array = np.array(mask)

        print(f"✓ Image size: {img.size}")
        print(f"✓ Image mode: {img.mode}")
        print(f"✓ Mask size: {mask.size}")
        print(f"✓ Mask mode: {mask.mode}")
        print(f"✓ Mask unique values: {sorted(np.unique(mask_array))}")
        print(f"✓ Mask value range: {mask_array.min()} - {mask_array.max()}")

        # Check if size is 512x512
        size_ok = img.size == (512, 512) and mask.size == (512, 512)
        print(f"✓ Size correct (512x512): {size_ok}")

        return size_ok

    except Exception as e:
        print(f"✗ Error loading images: {e}")
        return False

def main():
    print("CE7454 Face Parsing - Basic Test")

    # Test dataset structure
    structure_ok = test_dataset_structure()

    # Test image properties
    properties_ok = test_image_properties()

    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    print(f"Dataset structure: {'✓ PASS' if structure_ok else '✗ FAIL'}")
    print(f"Image properties: {'✓ PASS' if properties_ok else '✗ FAIL'}")

    if structure_ok and properties_ok:
        print("\n🎉 Basic tests passed!")
        print("Dataset is ready for training!")
        print("\nNext steps:")
        print("1. Ensure PyTorch and dependencies are installed")
        print("2. Run: python src/train_full.py")
    else:
        print("\n❌ Some basic tests failed. Please check dataset.")

if __name__ == "__main__":
    main()