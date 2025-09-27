"""
Full training script for CE7454 Face Parsing Project
Uses all 1000 training images without validation split
"""
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import time
from tqdm import tqdm

from config import Config
from dataset import get_dataloader, verify_dataset
from models.simple_unet import get_simple_model, count_parameters
from utils import CombinedLoss, calculate_f_score, pixel_accuracy, mean_iou
from utils import save_checkpoint, create_experiment_dir

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_f_score = 0.0
    running_pixel_acc = 0.0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
    for batch_idx, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        # Calculate metrics
        with torch.no_grad():
            pred = torch.argmax(outputs, dim=1)
            f_score = calculate_f_score(pred.cpu().numpy(), masks.cpu().numpy())
            pixel_acc = pixel_accuracy(pred, masks)

        running_loss += loss.item()
        running_f_score += f_score
        running_pixel_acc += pixel_acc

        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'F-Score': f'{f_score:.4f}',
            'PixelAcc': f'{pixel_acc:.4f}'
        })

    num_batches = len(dataloader)
    return {
        'loss': running_loss / num_batches,
        'f_score': running_f_score / num_batches,
        'pixel_acc': running_pixel_acc / num_batches
    }

def main():
    """Main training function"""
    print("="*60)
    print("CE7454 Face Parsing - Full Training (1000 images)")
    print("="*60)

    # Create experiment directory
    exp_dir = create_experiment_dir()
    print(f"Experiment directory: {exp_dir}")

    # Create necessary directories
    Config.create_dirs()

    # Verify dataset
    print("\nVerifying dataset...")
    verify_dataset(Config.DATA_ROOT)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Create data loader (only training, no validation)
    print("\nCreating data loader...")
    try:
        train_loader = get_dataloader(Config.DATA_ROOT, split='train', batch_size=Config.BATCH_SIZE)
        print(f"Training batches: {len(train_loader)}")
        print(f"Total training samples: {len(train_loader.dataset)}")
    except Exception as e:
        print(f"Error creating data loader: {e}")
        print("Please ensure dataset is properly structured.")
        return

    # Create model
    print("\nCreating model...")
    model = get_simple_model('simple_unet', n_classes=Config.NUM_CLASSES)
    model = model.to(device)

    # Verify parameter count
    param_count = count_parameters(model)
    if param_count >= Config.MAX_PARAMS:
        print(f"WARNING: Model has {param_count} parameters, exceeding limit of {Config.MAX_PARAMS}")
        return

    print(f"Model parameters: {param_count:,} (within limit: {param_count < Config.MAX_PARAMS})")

    # Loss function
    criterion = CombinedLoss(ce_weight=1.0, dice_weight=0.5)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

    # Training history
    train_history = {'loss': [], 'f_score': [], 'pixel_acc': []}

    best_f_score = 0.0
    best_model_path = os.path.join(exp_dir, 'best_model.pth')
    final_model_path = os.path.join(exp_dir, 'final_model.pth')

    print(f"\nStarting training for {Config.NUM_EPOCHS} epochs...")
    print("="*60)

    start_time = time.time()

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        print("-" * 40)

        # Training
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_history['loss'].append(train_metrics['loss'])
        train_history['f_score'].append(train_metrics['f_score'])
        train_history['pixel_acc'].append(train_metrics['pixel_acc'])

        # Learning rate scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print epoch results
        print(f"Train - Loss: {train_metrics['loss']:.4f}, F-Score: {train_metrics['f_score']:.4f}, PixelAcc: {train_metrics['pixel_acc']:.4f}")
        print(f"LR: {current_lr:.6f}")

        # Save best model based on F-Score
        if train_metrics['f_score'] > best_f_score:
            best_f_score = train_metrics['f_score']
            save_checkpoint(model, optimizer, epoch, train_metrics['loss'], best_model_path)
            print(f"✓ New best model saved! F-Score: {best_f_score:.4f}")

        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(exp_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, train_metrics['loss'], checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}")

    # Save final model
    save_checkpoint(model, optimizer, Config.NUM_EPOCHS-1, train_history['loss'][-1], final_model_path)

    # Training completed
    end_time = time.time()
    training_time = end_time - start_time

    print("\n" + "="*60)
    print("Training Completed!")
    print(f"Training time: {training_time/3600:.2f} hours")
    print(f"Best training F-Score: {best_f_score:.4f}")
    print(f"Final training F-Score: {train_history['f_score'][-1]:.4f}")
    print(f"Best model saved at: {best_model_path}")
    print(f"Final model saved at: {final_model_path}")
    print("="*60)

    # Save training history
    import pickle
    history_path = os.path.join(exp_dir, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump({'train': train_history}, f)

    print(f"\nNext steps:")
    print(f"1. Use best model for inference: {best_model_path}")
    print(f"2. Generate test predictions: python src/inference.py --model_path {best_model_path}")

    return best_f_score, best_model_path

if __name__ == "__main__":
    main()