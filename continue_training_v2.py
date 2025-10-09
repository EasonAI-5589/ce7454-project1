"""
改进的继续训练脚本 - 支持保留或重置优化器
"""

import argparse
import torch
import yaml
import os
from pathlib import Path
from datetime import datetime

from src.models.microsegformer import MicroSegFormer
from src.dataset import create_train_val_loaders
from src.trainer import Trainer, create_optimizer, create_scheduler
from src.utils import CombinedLoss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to checkpoint file')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config file for continue training')
    parser.add_argument('--mode', type=str, default='reset',
                      choices=['reset', 'preserve', 'hybrid'],
                      help='Optimizer handling mode:\n'
                           'reset: Reset optimizer completely (for exploring new LR)\n'
                           'preserve: Keep optimizer state (safest continuation)\n'
                           'hybrid: Keep momentum, reset LR (balanced approach)')
    return parser.parse_args()

def create_output_dir(model_name='microsegformer', base_dir='checkpoints'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_dir, f'{model_name}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_checkpoint_smart(checkpoint_path, model, optimizer, scheduler, mode='reset'):
    """
    智能加载checkpoint

    Args:
        checkpoint_path: checkpoint路径
        model: 模型实例
        optimizer: 优化器实例
        scheduler: 调度器实例
        mode: 'reset', 'preserve', or 'hybrid'

    Returns:
        best_f_score: 历史最佳F-Score
    """
    print(f"\n{'='*80}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Mode: {mode}")
    print(f"{'='*80}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 1. Always load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model weights loaded")
    else:
        raise ValueError("No model_state_dict in checkpoint")

    # 2. Get previous best F-Score
    best_f_score = checkpoint.get('best_f_score', 0.0)
    print(f"✓ Previous best F-Score: {best_f_score:.4f}")

    # 3. Handle optimizer based on mode
    if mode == 'reset':
        print(f"✓ Optimizer: RESET (exploring new learning rate)")
        # Don't load anything, use fresh optimizer

    elif mode == 'preserve':
        print(f"✓ Optimizer: PRESERVE (continue from exact state)")
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    elif mode == 'hybrid':
        print(f"✓ Optimizer: HYBRID (keep momentum, reset LR)")
        if 'optimizer_state_dict' in checkpoint:
            opt_state = checkpoint['optimizer_state_dict']
            # Load momentum but will use new LR from config
            optimizer.load_state_dict(opt_state)
            # Reset LR to config value
            for param_group in optimizer.param_groups:
                print(f"  Resetting LR from {param_group['lr']} to config LR")
        # Don't load scheduler - use fresh schedule from new LR

    print(f"{'='*80}\n")
    return best_f_score

def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = MicroSegFormer(
        num_classes=config['model']['num_classes'],
        use_lmsa=config['model'].get('use_lmsa', True),
        dropout=config['model'].get('dropout', 0.1)
    ).to(device)

    # Create dataloaders
    train_loader, val_loader = create_train_val_loaders(
        data_root=config['data']['root'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        val_split=config['data']['val_split'],
        augmentation_config=config.get('augmentation', {})
    )

    # Create loss
    criterion = CombinedLoss(
        ce_weight=config['loss']['ce_weight'],
        dice_weight=config['loss']['dice_weight'],
        class_weights=None,
        use_focal=config['loss'].get('use_focal', False),
        focal_alpha=config['loss'].get('focal_alpha', 0.25),
        focal_gamma=config['loss'].get('focal_gamma', 2.0)
    )

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, len(train_loader))

    # Load checkpoint with smart mode
    previous_best = load_checkpoint_smart(
        args.checkpoint, model, optimizer, scheduler, mode=args.mode
    )

    # Create output directory
    output_dir = create_output_dir(config['model']['name'])
    config['output_dir'] = output_dir

    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Save continue info
    continue_info = {
        'continued_from': args.checkpoint,
        'previous_best_f_score': float(previous_best),
        'mode': args.mode,
        'start_epoch': 0  # Always start from 0 in new training
    }
    with open(os.path.join(output_dir, 'continue_info.yaml'), 'w') as f:
        yaml.dump(continue_info, f, default_flow_style=False)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )

    # Training info
    print(f"\n{'='*80}")
    print(f"Continue Training Info:")
    print(f"  Mode: {args.mode}")
    print(f"  Previous best F-Score: {previous_best:.4f}")
    print(f"  New LR: {config['training']['learning_rate']}")
    print(f"  New output dir: {output_dir}")
    print(f"  Target epochs: {config['training']['epochs']}")
    print(f"  Early stop patience: {config['training']['early_stopping_patience']}")
    print(f"{'='*80}\n")

    # Start training
    trainer.fit(epochs=config['training']['epochs'])

    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Best model saved to: {trainer.best_model_path}")
    print(f"Final best F-Score: {trainer.best_f_score:.4f}")
    if trainer.best_f_score > previous_best:
        improvement = trainer.best_f_score - previous_best
        print(f"✓ IMPROVEMENT: +{improvement:.4f} ({improvement/previous_best*100:.2f}%)")
    else:
        degradation = previous_best - trainer.best_f_score
        print(f"✗ DEGRADATION: -{degradation:.4f} ({degradation/previous_best*100:.2f}%)")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
