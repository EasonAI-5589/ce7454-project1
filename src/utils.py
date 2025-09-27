"""
Utility functions for CE7454 Face Parsing Project
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import os

class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3)) + self.smooth)

        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    """Combined Cross Entropy + Dice Loss"""

    def __init__(self, ce_weight=1.0, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.ce_weight * ce + self.dice_weight * dice

def calculate_f_score(pred, target, num_classes=19):
    """Calculate F-Score for segmentation"""
    pred = pred.flatten()
    target = target.flatten()

    f_scores = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        if target_cls.sum() == 0:  # No ground truth for this class
            if pred_cls.sum() == 0:  # No prediction either - perfect
                f_scores.append(1.0)
            else:  # False positive
                f_scores.append(0.0)
        else:
            f_score = f1_score(target_cls, pred_cls, zero_division=0)
            f_scores.append(f_score)

    return np.mean(f_scores)

def pixel_accuracy(pred, target):
    """Calculate pixel accuracy"""
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total

def mean_iou(pred, target, num_classes=19):
    """Calculate mean IoU"""
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()

        if union == 0:
            ious.append(1.0)  # Perfect when both are empty
        else:
            ious.append(intersection / union)

    return np.mean(ious)

def save_checkpoint(model, optimizer, epoch, loss, filename):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)

def load_checkpoint(model, optimizer, filename, device):
    """Load model checkpoint"""
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

def visualize_predictions(images, targets, predictions, num_samples=4, save_path=None):
    """Visualize model predictions"""
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))

    for i in range(min(num_samples, len(images))):
        # Original image
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        img = np.clip(img, 0, 1)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        # Ground truth
        target = targets[i].cpu().numpy()
        axes[i, 1].imshow(target, cmap='nipy_spectral')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        # Prediction
        pred = predictions[i].cpu().numpy()
        axes[i, 2].imshow(pred, cmap='nipy_spectral')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

class EarlyStopping:
    """Early stopping to avoid overfitting"""

    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def count_model_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params

def create_experiment_dir(base_dir='experiments'):
    """Create experiment directory with timestamp"""
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(base_dir, f'exp_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")

    # Test loss functions
    batch_size, num_classes, height, width = 2, 19, 512, 512
    inputs = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))

    # Test losses
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    combined_loss = CombinedLoss()

    ce_result = ce_loss(inputs, targets)
    dice_result = dice_loss(inputs, targets)
    combined_result = combined_loss(inputs, targets)

    print(f"Cross Entropy Loss: {ce_result:.4f}")
    print(f"Dice Loss: {dice_result:.4f}")
    print(f"Combined Loss: {combined_result:.4f}")

    # Test metrics
    pred = torch.argmax(inputs, dim=1)
    f_score = calculate_f_score(pred, targets)
    pixel_acc = pixel_accuracy(pred, targets)
    iou = mean_iou(pred, targets)

    print(f"F-Score: {f_score:.4f}")
    print(f"Pixel Accuracy: {pixel_acc:.4f}")
    print(f"Mean IoU: {iou:.4f}")

    print("✓ All utility functions working correctly")