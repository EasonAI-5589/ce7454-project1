"""
继续训练脚本 - 从已有的 checkpoint 继续训练

用法:
    python continue_training.py \
        --checkpoint checkpoints/microsegformer_20251008_025917/best_model.pth \
        --config configs/lmsa_continue_from_best.yaml \
        --reset_optimizer  # 可选: 重置优化器状态
"""

import argparse
import torch
import yaml
import os
from pathlib import Path

from src.models.microsegformer import MicroSegFormer
from src.dataset import create_train_val_loaders
from src.loss import SegmentationLoss
from src.trainer import Trainer
from src.utils import create_optimizer, create_scheduler, create_output_dir

def load_checkpoint_for_continue(checkpoint_path, model, optimizer=None, scheduler=None, reset_optimizer=False):
    """
    加载checkpoint用于继续训练
    
    Args:
        checkpoint_path: checkpoint文件路径
        model: 模型实例
        optimizer: 优化器实例 (如果reset_optimizer=True则不加载)
        scheduler: 学习率调度器 (如果reset_optimizer=True则不加载)
        reset_optimizer: 是否重置优化器状态
    
    Returns:
        start_epoch: 开始的epoch数
        best_f_score: 历史最佳F-Score
    """
    print(f"\n{'='*70}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"{'='*70}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded model weights")
    
    # 获取checkpoint信息
    start_epoch = checkpoint.get('epoch', 0)
    best_f_score = checkpoint.get('best_f_score', 0.0)
    
    print(f"\nCheckpoint info:")
    print(f"  Epoch: {start_epoch}")
    print(f"  Best F-Score: {best_f_score:.6f}")
    
    if reset_optimizer:
        print(f"\n⚠️  RESET OPTIMIZER - Starting with fresh optimizer state")
        print(f"   This is recommended when:")
        print(f"   - Training stopped due to overfitting")
        print(f"   - Want to escape local minimum")
        print(f"   - Changed learning rate significantly")
        start_epoch = 0  # 重新计数epoch
    else:
        # 加载优化器和调度器状态
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✅ Loaded optimizer state")
        
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"✅ Loaded scheduler state")
    
    print(f"{'='*70}\n")
    
    return start_epoch, best_f_score


def main():
    parser = argparse.ArgumentParser(description='Continue training from checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--reset_optimizer', action='store_true', 
                        help='Reset optimizer state (recommended for escaping overfitting)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build model
    print("\n" + "="*70)
    print("Building model...")
    print("="*70)
    
    model = MicroSegFormer(
        num_classes=config['model']['num_classes'],
        dropout=config['model'].get('dropout', 0.0),
        use_lmsa=config['model'].get('use_lmsa', False)
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_limit = 1821085
    
    print(f"\nModel: {config['model']['name']}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Parameter usage: {trainable_params/param_limit*100:.1f}%")
    
    # Get dataloaders
    print("\nLoading dataset...")
    train_loader, val_loader = create_train_val_loaders(
        data_root=config['data']['root'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        val_split=config['data']['val_split']
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Loss function
    criterion = SegmentationLoss(
        num_classes=config['model']['num_classes'],
        ce_weight=config['loss']['ce_weight'],
        dice_weight=config['loss']['dice_weight'],
        device=device,
        use_focal=config['loss'].get('use_focal', False)
    )
    
    # Optimizer and scheduler
    optimizer = create_optimizer(model, config['training'])
    scheduler = create_scheduler(optimizer, config['training'], config['training']['epochs'])
    
    # Load checkpoint
    start_epoch, best_f_score = load_checkpoint_for_continue(
        checkpoint_path=args.checkpoint,
        model=model,
        optimizer=optimizer if not args.reset_optimizer else None,
        scheduler=scheduler if not args.reset_optimizer else None,
        reset_optimizer=args.reset_optimizer
    )
    
    # Create output directory
    output_dir = create_output_dir(model_name=config['model']['name'])
    config['output_dir'] = output_dir
    
    # Save config
    config_path = os.path.join(output_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Save continue info
    continue_info = {
        'continued_from': args.checkpoint,
        'reset_optimizer': args.reset_optimizer,
        'start_epoch': start_epoch,
        'previous_best_f_score': best_f_score
    }
    continue_info_path = os.path.join(output_dir, 'continue_info.yaml')
    with open(continue_info_path, 'w') as f:
        yaml.dump(continue_info, f, default_flow_style=False)
    
    print(f"\nConfig saved to: {config_path}")
    print(f"Continue info saved to: {continue_info_path}")
    
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
    
    # Override best f-score if not resetting
    if not args.reset_optimizer:
        trainer.best_f_score = best_f_score
        trainer.current_epoch = start_epoch
    
    print(f"\nStarting continued training: {config['experiment']['name']}")
    print(f"Description: {config['experiment']['description']}")
    print("="*70)
    
    # Train
    remaining_epochs = config['training']['epochs'] - start_epoch
    print(f"\nTraining for {remaining_epochs} more epochs (epoch {start_epoch} -> {config['training']['epochs']})")
    
    trainer.fit(epochs=config['training']['epochs'], start_epoch=start_epoch)
    
    print("\n" + "="*70)
    print("Continued training completed!")
    print(f"Results saved to: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
