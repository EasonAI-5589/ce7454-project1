"""
Test data loading for CE7454 Face Parsing Project
"""
import sys
sys.path.append('src')

from config import Config
from dataset import verify_dataset, get_dataloader
from models.simple_unet import get_simple_model

def test_data_loading():
    """Test if data loading works correctly"""
    print("="*50)
    print("Testing Data Loading")
    print("="*50)

    # Verify dataset structure
    verify_dataset(Config.DATA_ROOT)

    # Test data loader
    print("\nTesting data loader...")
    try:
        train_loader = get_dataloader(Config.DATA_ROOT, split='train', batch_size=2)
        print(f"✓ Training loader created: {len(train_loader)} batches")
        print(f"✓ Dataset size: {len(train_loader.dataset)} samples")

        # Test loading one batch
        print("\nTesting batch loading...")
        for i, (images, masks) in enumerate(train_loader):
            print(f"✓ Batch {i+1}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Masks shape: {masks.shape}")
            print(f"  Images dtype: {images.dtype}")
            print(f"  Masks dtype: {masks.dtype}")
            print(f"  Masks min/max: {masks.min()}/{masks.max()}")

            if i == 0:  # Only test first batch
                break

        print("✓ Data loading test successful!")

    except Exception as e:
        print(f"✗ Data loading error: {e}")
        return False

    return True

def test_model():
    """Test if model can process the data"""
    print("\n" + "="*50)
    print("Testing Model")
    print("="*50)

    try:
        # Create model
        model = get_simple_model('simple_unet', n_classes=Config.NUM_CLASSES)
        print(f"✓ Model created successfully")

        # Test forward pass with dummy data
        import torch
        dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)

        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        print(f"✓ Forward pass successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected output shape: (2, {Config.NUM_CLASSES}, {Config.IMAGE_SIZE}, {Config.IMAGE_SIZE})")

        if output.shape == (2, Config.NUM_CLASSES, Config.IMAGE_SIZE, Config.IMAGE_SIZE):
            print("✓ Output shape correct!")
            return True
        else:
            print("✗ Output shape incorrect!")
            return False

    except Exception as e:
        print(f"✗ Model test error: {e}")
        return False

if __name__ == "__main__":
    print("CE7454 Face Parsing - Data Loading Test")

    # Test data loading
    data_ok = test_data_loading()

    # Test model
    model_ok = test_model()

    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    print(f"Data loading: {'✓ PASS' if data_ok else '✗ FAIL'}")
    print(f"Model test: {'✓ PASS' if model_ok else '✗ FAIL'}")

    if data_ok and model_ok:
        print("\n🎉 All tests passed! Ready to start training!")
        print("Run: python src/train_full.py")
    else:
        print("\n❌ Some tests failed. Please fix issues before training.")