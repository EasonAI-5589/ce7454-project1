#!/usr/bin/env python3
"""
Main training script for MicroSegFormer
CE7454 Face Parsing Project
"""

import argparse
import yaml
import torch
import os
from datetime import datetime

from src.trainer import Trainer, create_optimizer, create_scheduler
from src.dataset import create_train_val_loaders
from src.models.microsegformer import MicroSegFormer
from src.utils import CombinedLoss


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model(config):
    """Initialize model based on config"""
    model_name = config['model']['name']
    num_classes = config['model']['num_classes']
    dropout = config['model'].get('dropout', 0.0)

    if model_name == 'microsegformer':
        model = MicroSegFormer(num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def create_output_dir(base_dir='checkpoints'):
    """Create timestamped output directory"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_dir, f'microsegformer_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


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

    # Create data loaders
    print("\nLoading dataset...")
    train_loader, val_loader = create_train_val_loaders(
        data_root=config['data']['root'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        val_split=config['data']['val_split']
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create loss function with class weights
    class_weights = None
    if config['loss'].get('use_class_weights', False):
        # Class weights calculated from training data distribution
        class_weights = torch.tensor([
            0.0043, 0.0049, 0.0602, 0.3834, 0.5577,
            0.5577, 0.2924, 0.2986, 0.2595, 0.3189,
            0.4357, 0.3030, 0.1857, 0.0040, 0.1121,
            0.6114, 14.5434, 0.0305, 0.0366
        ], device=device)

    criterion = CombinedLoss(
        ce_weight=config['loss']['ce_weight'],
        dice_weight=config['loss']['dice_weight'],
        class_weights=class_weights,
        use_focal=config['loss'].get('use_focal', False),
        focal_alpha=config['loss'].get('focal_alpha', 0.25),
        focal_gamma=config['loss'].get('focal_gamma', 2.0)
    )

    # Create optimizer
    optimizer = create_optimizer(model, config['training'])

    # Create scheduler
    scheduler = create_scheduler(optimizer, config['training'], config['training']['epochs'])

    # Create output directory
    output_dir = create_output_dir()
    config['output_dir'] = output_dir

    print(f"\nOutput directory: {output_dir}")

    # Save config
    config_path = os.path.join(output_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config saved to: {config_path}")

    # Setup trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device
    )

    # Load checkpoint if resuming
    if args.resume:
        print(f"\nLoading checkpoint from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resumed from epoch {checkpoint['epoch']}")

    # Start training
    print(f"\nStarting training: {config['experiment']['name']}")
    print(f"Description: {config['experiment']['description']}")
    print("=" * 80)

    trainer.fit(epochs=config['training']['epochs'])

    print("\n✓ Training completed!")
    print(f"Best model saved at: {trainer.best_model_path}")


if __name__ == '__main__':
    main()
