"""
Test script to verify model implementations and basic functionality
"""
import torch
import sys
import os

# Add src to path
sys.path.append('src')

from models.unet import get_model, count_parameters
from config import Config
from utils import CombinedLoss, calculate_f_score, pixel_accuracy

def test_models():
    """Test different model architectures"""
    print("="*60)
    print("Testing Model Architectures")
    print("="*60)

    models_to_test = ['unet', 'resnet_unet', 'resnet18_unet']

    for model_name in models_to_test:
        print(f"\n--- Testing {model_name} ---")

        try:
            model = get_model(model_name, n_classes=Config.NUM_CLASSES)
            param_count = count_parameters(model)

            print(f"✓ Model created successfully")
            print(f"✓ Parameters: {param_count:,}")
            print(f"✓ Within limit: {param_count < Config.MAX_PARAMS}")

            # Test forward pass
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)

            # Create dummy input
            batch_size = 2
            x = torch.randn(batch_size, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(device)

            model.eval()
            with torch.no_grad():
                y = model(x)

            expected_shape = (batch_size, Config.NUM_CLASSES, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
            print(f"✓ Input shape: {tuple(x.shape)}")
            print(f"✓ Output shape: {tuple(y.shape)}")
            print(f"✓ Expected shape: {expected_shape}")
            print(f"✓ Shape correct: {tuple(y.shape) == expected_shape}")

            if param_count >= Config.MAX_PARAMS:
                print(f"⚠  WARNING: {model_name} exceeds parameter limit!")

        except Exception as e:
            print(f"✗ Error testing {model_name}: {str(e)}")

def test_loss_functions():
    """Test loss function implementations"""
    print("\n" + "="*60)
    print("Testing Loss Functions")
    print("="*60)

    batch_size, num_classes, H, W = 2, Config.NUM_CLASSES, 64, 64

    # Create dummy data
    inputs = torch.randn(batch_size, num_classes, H, W)
    targets = torch.randint(0, num_classes, (batch_size, H, W))

    print(f"Input shape: {tuple(inputs.shape)}")
    print(f"Target shape: {tuple(targets.shape)}")

    try:
        # Test Cross Entropy Loss
        ce_loss = torch.nn.CrossEntropyLoss()
        ce_result = ce_loss(inputs, targets)
        print(f"✓ CrossEntropyLoss: {ce_result:.4f}")

        # Test Combined Loss
        combined_loss = CombinedLoss()
        combined_result = combined_loss(inputs, targets)
        print(f"✓ CombinedLoss: {combined_result:.4f}")

    except Exception as e:
        print(f"✗ Error testing loss functions: {str(e)}")

def test_metrics():
    """Test metric calculations"""
    print("\n" + "="*60)
    print("Testing Metrics")
    print("="*60)

    # Create dummy predictions and targets
    batch_size, H, W = 2, 64, 64
    pred = torch.randint(0, Config.NUM_CLASSES, (batch_size, H, W))
    target = torch.randint(0, Config.NUM_CLASSES, (batch_size, H, W))

    print(f"Prediction shape: {tuple(pred.shape)}")
    print(f"Target shape: {tuple(target.shape)}")

    try:
        # Test F-Score
        f_score = calculate_f_score(pred.numpy(), target.numpy(), Config.NUM_CLASSES)
        print(f"✓ F-Score: {f_score:.4f}")

        # Test Pixel Accuracy
        pixel_acc = pixel_accuracy(pred, target)
        print(f"✓ Pixel Accuracy: {pixel_acc:.4f}")

    except Exception as e:
        print(f"✗ Error testing metrics: {str(e)}")

def test_data_loading():
    """Test data loading capabilities"""
    print("\n" + "="*60)
    print("Testing Data Loading")
    print("="*60)

    try:
        from dataset import verify_dataset, get_transforms

        # Verify dataset structure
        print("Dataset structure verification:")
        verify_dataset(Config.DATA_ROOT)

        # Test transforms
        print(f"\n✓ Training transforms created")
        train_transform = get_transforms('train')

        print(f"✓ Validation transforms created")
        val_transform = get_transforms('val')

        # Create dummy PIL image to test transforms
        from PIL import Image
        dummy_image = Image.new('RGB', (Config.IMAGE_SIZE, Config.IMAGE_SIZE))

        train_output = train_transform(dummy_image)
        val_output = val_transform(dummy_image)

        print(f"✓ Train transform output shape: {tuple(train_output.shape)}")
        print(f"✓ Val transform output shape: {tuple(val_output.shape)}")

    except Exception as e:
        print(f"✗ Error testing data loading: {str(e)}")

def test_config():
    """Test configuration settings"""
    print("\n" + "="*60)
    print("Testing Configuration")
    print("="*60)

    print(f"✓ Number of classes: {Config.NUM_CLASSES}")
    print(f"✓ Image size: {Config.IMAGE_SIZE}")
    print(f"✓ Max parameters: {Config.MAX_PARAMS:,}")
    print(f"✓ Batch size: {Config.BATCH_SIZE}")
    print(f"✓ Learning rate: {Config.LEARNING_RATE}")
    print(f"✓ Number of epochs: {Config.NUM_EPOCHS}")

    # Test directory creation
    try:
        Config.create_dirs()
        print("✓ Directories created successfully")
    except Exception as e:
        print(f"✗ Error creating directories: {str(e)}")

def main():
    """Run all tests"""
    print("CE7454 Face Parsing - Model and Functionality Tests")

    # Set device info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Run tests
    test_config()
    test_models()
    test_loss_functions()
    test_metrics()
    test_data_loading()

    print("\n" + "="*60)
    print("All Tests Completed!")
    print("="*60)

if __name__ == "__main__":
    main()