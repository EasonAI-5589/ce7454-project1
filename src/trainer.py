"""
Trainer class for face parsing model training
Supports early stopping, learning rate warmup, and validation
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import os
import time
import json
from collections import defaultdict
import numpy as np

from .utils import AverageMeter, calculate_f_score, pixel_accuracy, save_checkpoint


class Trainer:
    """
    Trainer class for model training with validation and early stopping
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        config,
        device='cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        # Training state
        self.current_epoch = 0
        self.best_f_score = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

        # History
        self.history = defaultdict(list)

        # Paths
        self.output_dir = config.get('output_dir', 'outputs/default')
        os.makedirs(self.output_dir, exist_ok=True)

        self.best_model_path = os.path.join(self.output_dir, 'best_model.pth')
        self.last_model_path = os.path.join(self.output_dir, 'last_model.pth')

        # Early stopping (can be None to disable)
        self.early_stopping_patience = config.get('early_stopping_patience', 20)
        self.use_early_stopping = self.early_stopping_patience is not None

        # Gradient clipping
        self.max_grad_norm = config.get('max_grad_norm', 1.0)

        # Mixed precision (optional)
        self.use_amp = config.get('use_amp', False)
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()

        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Output dir: {self.output_dir}")
        if self.use_early_stopping:
            print(f"  Early stopping patience: {self.early_stopping_patience}")
        else:
            print(f"  Early stopping: DISABLED (will train full epochs)")
        print(f"  Max grad norm: {self.max_grad_norm}")
        print(f"  Mixed precision: {self.use_amp}")

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()

        losses = AverageMeter()
        f_scores = AverageMeter()
        accuracies = AverageMeter()

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                # Backward with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                # Backward
                loss.backward()

                # Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.optimizer.step()

            # Calculate metrics
            with torch.no_grad():
                pred = torch.argmax(outputs, dim=1)
                f_score = calculate_f_score(pred.cpu().numpy(), masks.cpu().numpy())
                accuracy = pixel_accuracy(pred, masks)

            # Update meters
            # Note: f_score is already averaged across batch, so weight=1
            losses.update(loss.item(), images.size(0))
            f_scores.update(f_score, 1)  # Fixed: f_score is mean, not sum
            accuracies.update(accuracy, images.size(0))

            # Print progress
            if batch_idx % self.config.get('print_every', 10) == 0:
                print(f'  [{batch_idx:3d}/{len(self.train_loader):3d}] '
                      f'Loss: {losses.avg:.4f} F-Score: {f_scores.avg:.4f} Acc: {accuracies.avg:.4f}')

        return {
            'loss': losses.avg,
            'f_score': f_scores.avg,
            'accuracy': accuracies.avg
        }

    @torch.no_grad()
    def validate(self):
        """Validate on validation set"""
        self.model.eval()

        losses = AverageMeter()
        f_scores = AverageMeter()
        accuracies = AverageMeter()

        for images, masks in self.val_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # Calculate metrics
            pred = torch.argmax(outputs, dim=1)
            f_score = calculate_f_score(pred.cpu().numpy(), masks.cpu().numpy())
            accuracy = pixel_accuracy(pred, masks)

            # Update meters
            # Note: f_score is already averaged across batch, so weight=1
            losses.update(loss.item(), images.size(0))
            f_scores.update(f_score, 1)  # Fixed: f_score is mean, not sum
            accuracies.update(accuracy, images.size(0))

        return {
            'loss': losses.avg,
            'f_score': f_scores.avg,
            'accuracy': accuracies.avg
        }

    def check_early_stop(self, val_f_score):
        """Check if should early stop"""
        # If early stopping is disabled, never stop
        if not self.use_early_stopping:
            # Still track best score
            if val_f_score > self.best_f_score:
                self.best_f_score = val_f_score
                self.best_epoch = self.current_epoch
            return False

        # Normal early stopping logic
        if val_f_score > self.best_f_score:
            self.best_f_score = val_f_score
            self.best_epoch = self.current_epoch
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered! No improvement for {self.early_stopping_patience} epochs.")
                print(f"Best F-Score: {self.best_f_score:.4f} at epoch {self.best_epoch}")
                return True
            return False

    def save_checkpoint(self, filepath, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_f_score': self.best_f_score,
            'config': self.config,
            'history': dict(self.history)
        }

        torch.save(checkpoint, filepath)

        if is_best:
            print(f"  ✓ New best model saved! F-Score: {self.best_f_score:.4f}")

    def fit(self, epochs):
        """Complete training loop"""
        print(f"\nStarting training for {epochs} epochs...")
        print("=" * 80)

        start_time = time.time()

        for epoch in range(epochs):
            self.current_epoch = epoch

            print(f"\nEpoch {epoch+1}/{epochs}")

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_f_score'].append(train_metrics['f_score'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_f_score'].append(val_metrics['f_score'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['lr'].append(current_lr)

            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"F-Score: {train_metrics['f_score']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"F-Score: {val_metrics['f_score']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            print(f"  LR: {current_lr:.6f}")

            # Save best model
            if val_metrics['f_score'] > self.best_f_score:
                self.save_checkpoint(self.best_model_path, is_best=True)

            # Save last model
            self.save_checkpoint(self.last_model_path, is_best=False)

            # Early stopping check
            if self.check_early_stop(val_metrics['f_score']):
                break

        # Training completed
        end_time = time.time()
        training_time = end_time - start_time

        print("\n" + "=" * 80)
        print("Training Completed!")
        print(f"Total time: {training_time/3600:.2f} hours")
        print(f"Best F-Score: {self.best_f_score:.4f} (Epoch {self.best_epoch+1})")
        print(f"Final F-Score: {self.history['val_f_score'][-1]:.4f}")
        print(f"Best model: {self.best_model_path}")
        print("=" * 80)

        # Save history
        self.save_history()

        return self.history

    def save_history(self):
        """Save training history to JSON"""
        history_path = os.path.join(self.output_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(dict(self.history), f, indent=2)
        print(f"Training history saved to: {history_path}")


def create_optimizer(model, config):
    """Create optimizer from config"""
    optimizer_type = config.get('optimizer', 'AdamW')
    lr = float(config.get('learning_rate', 1e-3))  # Force float conversion for YAML scientific notation
    weight_decay = float(config.get('weight_decay', 1e-4))

    if optimizer_type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        momentum = config.get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    return optimizer


def create_scheduler(optimizer, config, num_epochs):
    """Create learning rate scheduler with warmup"""
    scheduler_type = config.get('scheduler', 'CosineAnnealingLR')
    warmup_epochs = config.get('warmup_epochs', 5)

    if warmup_epochs > 0:
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )

    # Main scheduler
    if scheduler_type == 'CosineAnnealingLR':
        main_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
    elif scheduler_type == 'StepLR':
        step_size = config.get('step_size', 30)
        gamma = config.get('gamma', 0.1)
        from torch.optim.lr_scheduler import StepLR
        main_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        main_scheduler = None

    # Combine warmup and main scheduler
    if warmup_epochs > 0 and main_scheduler:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )
    elif main_scheduler:
        scheduler = main_scheduler
    else:
        scheduler = None

    return scheduler
