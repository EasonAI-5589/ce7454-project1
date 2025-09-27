"""
Quick test script to verify basic functionality without external dependencies
"""
import torch
import sys
import os
import numpy as np

# Add src to path
sys.path.append('src')

def test_simple_models():
    """Test simple model implementations"""
    print("="*60)
    print("Testing Simple Model Architectures")
    print("="*60)

    try:
        from models.simple_unet import get_simple_model, count_parameters

        models_to_test = ['simple_unet', 'lightweight_resnet_unet']

        for model_name in models_to_test:
            print(f"\n--- Testing {model_name} ---")

            model = get_simple_model(model_name, n_classes=19)
            param_count = count_parameters(model)

            print(f"✓ Model created successfully")
            print(f"✓ Parameters: {param_count:,}")
            print(f"✓ Within limit: {param_count < 1821085}")

            # Test forward pass
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)

            # Create dummy input
            batch_size = 2
            x = torch.randn(batch_size, 3, 512, 512).to(device)

            model.eval()
            with torch.no_grad():
                y = model(x)

            expected_shape = (batch_size, 19, 512, 512)
            print(f"✓ Input shape: {tuple(x.shape)}")
            print(f"✓ Output shape: {tuple(y.shape)}")
            print(f"✓ Expected shape: {expected_shape}")
            print(f"✓ Shape correct: {tuple(y.shape) == expected_shape}")

            if param_count >= 1821085:
                print(f"⚠  WARNING: {model_name} exceeds parameter limit!")

            # Test gradient computation
            model.train()
            x.requires_grad_(True)
            y = model(x)
            loss = y.mean()
            loss.backward()
            print(f"✓ Gradient computation successful")

    except Exception as e:
        print(f"✗ Error testing simple models: {str(e)}")
        import traceback
        traceback.print_exc()

def test_basic_operations():
    """Test basic PyTorch operations"""
    print("\n" + "="*60)
    print("Testing Basic Operations")
    print("="*60)

    try:
        # Test tensor operations
        x = torch.randn(2, 3, 512, 512)
        print(f"✓ Tensor creation: {tuple(x.shape)}")

        # Test convolution
        conv = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        y = conv(x)
        print(f"✓ Convolution: {tuple(y.shape)}")

        # Test loss functions
        ce_loss = torch.nn.CrossEntropyLoss()
        logits = torch.randn(2, 19, 64, 64)
        targets = torch.randint(0, 19, (2, 64, 64))
        loss = ce_loss(logits, targets)
        print(f"✓ CrossEntropyLoss: {loss:.4f}")

        # Test metrics
        pred = torch.argmax(logits, dim=1)
        accuracy = (pred == targets).float().mean()
        print(f"✓ Pixel Accuracy: {accuracy:.4f}")

    except Exception as e:
        print(f"✗ Error in basic operations: {str(e)}")

def test_config():
    """Test configuration"""
    print("\n" + "="*60)
    print("Testing Configuration")
    print("="*60)

    try:
        from config import Config

        print(f"✓ Number of classes: {Config.NUM_CLASSES}")
        print(f"✓ Image size: {Config.IMAGE_SIZE}")
        print(f"✓ Max parameters: {Config.MAX_PARAMS:,}")
        print(f"✓ Batch size: {Config.BATCH_SIZE}")
        print(f"✓ Learning rate: {Config.LEARNING_RATE}")

        Config.create_dirs()
        print("✓ Directories created successfully")

    except Exception as e:
        print(f"✗ Error testing config: {str(e)}")

def main():
    """Run quick tests"""
    print("CE7454 Face Parsing - Quick Test")

    # Device info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Run tests
    test_config()
    test_basic_operations()
    test_simple_models()

    print("\n" + "="*60)
    print("Quick Tests Completed!")
    print("="*60)

    print("\nNext steps:")
    print("1. Download CelebAMask-HQ mini dataset")
    print("2. Place dataset in 'data' directory")
    print("3. Run: python src/train.py")

if __name__ == "__main__":
    main()