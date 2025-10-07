#!/usr/bin/env python3
"""
Quick test script to verify model configurations
"""
import yaml
import torch
from src.models.microsegformer import MicroSegFormer
from src.models.unet import UNet


def test_model_from_config(config_path):
    """Test model initialization from config"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_name = config['model']['name']
    num_classes = config['model']['num_classes']
    dropout = config['model'].get('dropout', 0.0)

    print(f"\n{'='*70}")
    print(f"Testing: {config['experiment']['name']}")
    print(f"Config: {config_path}")
    print(f"{'='*70}")

    # Initialize model
    if model_name == 'microsegformer':
        use_lmsa = config['model'].get('use_lmsa', False)
        model = MicroSegFormer(num_classes=num_classes, dropout=dropout, use_lmsa=use_lmsa)
    elif model_name == 'unet':
        base_channels = config['model'].get('base_channels', 20)
        bilinear = config['model'].get('bilinear', True)
        model = UNet(n_channels=3, n_classes=num_classes, base_channels=base_channels,
                     bilinear=bilinear, dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_limit = 1821085
    usage_percent = (total_params / param_limit) * 100

    print(f"Model: {model_name}")
    if model_name == 'unet':
        print(f"  Base channels: {config['model']['base_channels']}")
        print(f"  Bilinear: {config['model']['bilinear']}")
    if model_name == 'microsegformer':
        print(f"  LMSA: {config['model'].get('use_lmsa', False)}")
    print(f"  Dropout: {dropout}")

    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Usage: {usage_percent:.1f}%")
    print(f"  Limit: {param_limit:,}")
    print(f"  Within limit: {'✓ YES' if total_params <= param_limit else '✗ NO'}")

    # Test forward pass
    print(f"\nForward pass test:")
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        y = model(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    assert y.shape == (1, 19, 512, 512), f"Output shape mismatch!"
    print(f"  ✓ PASSED")

    return total_params, usage_percent


def main():
    """Test all model configurations"""
    print("Testing Model Configurations")
    print("="*70)

    configs = [
        'configs/unet.yaml',
        'configs/lmsa.yaml',
    ]

    results = []
    for config_path in configs:
        try:
            params, usage = test_model_from_config(config_path)
            results.append((config_path, params, usage, '✓'))
        except Exception as e:
            print(f"✗ FAILED: {e}")
            results.append((config_path, 0, 0, '✗'))

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<30} {'Parameters':>12} {'Usage':>8} {'Status':>6}")
    print(f"{'-'*70}")
    for config, params, usage, status in results:
        config_name = config.split('/')[-1]
        print(f"{config_name:<30} {params:>12,} {usage:>7.1f}% {status:>6}")

    print(f"\n✓ All tests completed!")


if __name__ == '__main__':
    main()
