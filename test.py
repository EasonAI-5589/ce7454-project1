"""
Test all components before training
"""
import sys
import os

# Add src to path
sys.path.append('src')

def test_model():
    """Test model creation"""
    print("="*50)
    print("Testing Model")
    print("="*50)

    try:
        from model import test_model
        return test_model()
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_dataset():
    """Test dataset loading"""
    print("\n" + "="*50)
    print("Testing Dataset")
    print("="*50)

    try:
        from dataset import test_dataset
        return test_dataset()
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False

def test_utils():
    """Test utility functions"""
    print("\n" + "="*50)
    print("Testing Utils")
    print("="*50)

    try:
        from utils import test_utils
        test_utils()
        return True
    except Exception as e:
        print(f"✗ Utils test failed: {e}")
        return False

def check_requirements():
    """Check basic requirements"""
    print("="*50)
    print("Checking Requirements")
    print("="*50)

    requirements = [
        ('torch', 'PyTorch'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
    ]

    all_good = True
    for module, name in requirements:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - Install with: pip install {module}")
            all_good = False

    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("⚠ CUDA not available (will use CPU)")
    except:
        print("✗ Cannot check CUDA")

    return all_good

def check_data_structure():
    """Check dataset structure"""
    print("\n" + "="*50)
    print("Checking Data Structure")
    print("="*50)

    data_root = "data"
    required_dirs = [
        "data/train/images",
        "data/train/masks",
        "data/test/images"
    ]

    all_good = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            count = len([f for f in os.listdir(dir_path)
                        if f.endswith(('.jpg', '.png'))])
            print(f"✓ {dir_path}: {count} files")
        else:
            print(f"✗ {dir_path}: Not found")
            all_good = False

    if not all_good:
        print("\nTo fix data structure:")
        print("1. Download dev-public.zip from CodaBench")
        print("2. Extract to project root")
        print("3. Should create data/train/ and data/test/ folders")

    return all_good

def main():
    """Run all tests"""
    print("CE7454 Face Parsing - Component Tests")

    tests = [
        ("Requirements", check_requirements),
        ("Data Structure", check_data_structure),
        ("Model", test_model),
        ("Dataset", test_dataset),
        ("Utils", test_utils),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)

    all_passed = True
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<15}: {status}")
        if not result:
            all_passed = False

    print("\n" + "="*50)
    if all_passed:
        print("🎉 All tests passed! Ready to train!")
        print("Run: python src/train.py")
    else:
        print("❌ Some tests failed. Please fix issues before training.")
        print("Check error messages above for guidance.")

if __name__ == "__main__":
    main()