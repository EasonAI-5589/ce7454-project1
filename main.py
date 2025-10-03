#!/usr/bin/env python3
"""
Main training script for MicroSegFormer
CE7454 Face Parsing Project
"""

import argparse
import yaml
import torch
from pathlib import Path
from src.trainer import Trainer
from src.dataset import FaceParsingDataset
from src.models.microsegformer import MicroSegFormer


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model(config):
    """Initialize model based on config"""
    model_name = config['model']['name']
    num_classes = config['model']['num_classes']

    if model_name == 'microsegformer':
        model = MicroSegFormer(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Train face parsing model')
    parser.add_argument('--config', type=str, default='configs/main.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu), auto-detect if not specified')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    # Setup model
    model = setup_model(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_limit = 1821085
    usage_percent = (total_params / param_limit) * 100

    print(f"\nModel: {config['model']['name']}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Parameter usage: {usage_percent:.1f}%")
    print(f"Parameter limit: {param_limit:,}")

    if total_params > param_limit:
        raise ValueError(f"Model exceeds parameter limit! {total_params:,} > {param_limit:,}")

    # Setup trainer
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        resume_checkpoint=args.resume
    )

    # Start training
    print(f"\nStarting training: {config['experiment']['name']}")
    print(f"Description: {config['experiment']['description']}")
    print("-" * 80)

    trainer.train()

    print("\nTraining completed!")


if __name__ == '__main__':
    main()
