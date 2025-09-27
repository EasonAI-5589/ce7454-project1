"""
Face Parsing Training Script
Minimal dependencies, complete training pipeline
"""
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import time

from model import FaceParsingNet, count_parameters
from dataset import get_dataloader
from utils import CombinedLoss, calculate_f_score, pixel_accuracy
from utils import save_checkpoint, create_output_dir, AverageMeter

# Configuration
CONFIG = {
    'data_root': 'data',
    'batch_size': 8,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'num_epochs': 100,
    'num_classes': 19,
    'image_size': 512,
    'max_params': 1821085,
    'save_every': 20,
    'print_every': 10,
}

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()

    losses = AverageMeter()
    f_scores = AverageMeter()
    accuracies = AverageMeter()

    for batch_idx, (images, masks) in enumerate(dataloader):
        images, masks = images.to(device), masks.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate metrics
        with torch.no_grad():
            pred = torch.argmax(outputs, dim=1)
            f_score = calculate_f_score(pred.cpu().numpy(), masks.cpu().numpy())
            accuracy = pixel_accuracy(pred, masks)

        # Update meters
        losses.update(loss.item(), images.size(0))
        f_scores.update(f_score, images.size(0))
        accuracies.update(accuracy, images.size(0))

        # Print progress
        if batch_idx % CONFIG['print_every'] == 0:
            print(f'Epoch {epoch+1:3d} [{batch_idx:3d}/{len(dataloader):3d}] '
                  f'Loss: {losses.avg:.4f} F-Score: {f_scores.avg:.4f} Acc: {accuracies.avg:.4f}')

    return {
        'loss': losses.avg,
        'f_score': f_scores.avg,
        'accuracy': accuracies.avg
    }

def main():
    """Main training function"""
    print("="*80)
    print("CE7454 Face Parsing Training")
    print("="*80)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Create output directory
    output_dir = create_output_dir()
    print(f"Output directory: {output_dir}")

    # Create model
    print("\nCreating model...")
    model = FaceParsingNet(n_classes=CONFIG['num_classes'])
    param_count = count_parameters(model)

    print(f"Model parameters: {param_count:,}")
    print(f"Within limit: {param_count < CONFIG['max_params']}")

    if param_count >= CONFIG['max_params']:
        print(f"ERROR: Model exceeds parameter limit!")
        return

    model = model.to(device)

    # Create dataloader
    print("\nCreating dataloader...")
    try:
        train_loader = get_dataloader(
            CONFIG['data_root'],
            split='train',
            batch_size=CONFIG['batch_size'],
            augment=True
        )
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Training batches: {len(train_loader)}")

    except Exception as e:
        print(f"ERROR creating dataloader: {e}")
        print("Please ensure dataset is in 'data' directory with correct structure:")
        print("data/train/images/ and data/train/masks/")
        return

    # Setup training
    criterion = CombinedLoss(ce_weight=1.0, dice_weight=0.5)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])

    # Training history
    history = {
        'loss': [],
        'f_score': [],
        'accuracy': []
    }

    best_f_score = 0.0
    best_model_path = os.path.join(output_dir, 'best_model.pth')

    print(f"\nStarting training for {CONFIG['num_epochs']} epochs...")
    print("-"*80)

    start_time = time.time()

    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Update history
        history['loss'].append(train_metrics['loss'])
        history['f_score'].append(train_metrics['f_score'])
        history['accuracy'].append(train_metrics['accuracy'])

        # Learning rate step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print epoch summary
        print(f"Summary - Loss: {train_metrics['loss']:.4f} "
              f"F-Score: {train_metrics['f_score']:.4f} "
              f"Accuracy: {train_metrics['accuracy']:.4f} "
              f"LR: {current_lr:.6f}")

        # Save best model
        if train_metrics['f_score'] > best_f_score:
            best_f_score = train_metrics['f_score']
            save_checkpoint(model, optimizer, epoch, train_metrics['loss'], best_model_path)
            print(f"✓ New best model saved! F-Score: {best_f_score:.4f}")

        # Save checkpoint
        if (epoch + 1) % CONFIG['save_every'] == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, train_metrics['loss'], checkpoint_path)

    # Final model
    final_model_path = os.path.join(output_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, CONFIG['num_epochs']-1,
                   history['loss'][-1], final_model_path)

    # Training summary
    end_time = time.time()
    training_time = end_time - start_time

    print("\n" + "="*80)
    print("Training Completed!")
    print(f"Total time: {training_time/3600:.2f} hours")
    print(f"Best F-Score: {best_f_score:.4f}")
    print(f"Final F-Score: {history['f_score'][-1]:.4f}")
    print(f"Best model: {best_model_path}")
    print(f"Final model: {final_model_path}")

    # Save history
    import json
    history_path = os.path.join(output_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nNext steps:")
    print(f"1. Use model for inference: python src/inference.py --model {best_model_path}")
    print(f"2. Generate test predictions for Codabench submission")
    print("="*80)

if __name__ == "__main__":
    main()