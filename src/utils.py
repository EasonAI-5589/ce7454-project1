"""
Simple utilities for face parsing
Only depends on PyTorch and numpy
"""
import torch
import torch.nn as nn
import numpy as np
import os

def calculate_f_score(pred, target, num_classes=19, eps=1e-8):
    """Calculate F-Score for segmentation"""
    # Convert to numpy if tensor
    if hasattr(pred, 'cpu'):
        pred = pred.cpu().numpy()
    if hasattr(target, 'cpu'):
        target = target.cpu().numpy()

    pred = pred.flatten()
    target = target.flatten()

    f_scores = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        tp = float((pred_cls & target_cls).sum())
        fp = float((pred_cls & ~target_cls).sum())
        fn = float((~pred_cls & target_cls).sum())

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f_score = 2 * precision * recall / (precision + recall + eps)

        f_scores.append(f_score)

    return np.mean(f_scores)

def pixel_accuracy(pred, target):
    """Calculate pixel accuracy"""
    correct = (pred == target).sum().float()
    total = target.numel()
    return (correct / total).item()

class CombinedLoss(nn.Module):
    """Combined Cross Entropy + Dice Loss"""

    def __init__(self, ce_weight=1.0, dice_weight=0.5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def dice_loss(self, inputs, targets, eps=1e-8):
        """Dice loss"""
        inputs = torch.softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        dice = (2 * intersection + eps) / (
            inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3)) + eps
        )

        return 1 - dice.mean()

    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.ce_weight * ce + self.dice_weight * dice

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)

def load_checkpoint(model, optimizer, filepath, device):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

def create_output_dir(base_dir='outputs'):
    """Create timestamped output directory"""
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_dir, f'exp_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def test_utils():
    """Test utility functions"""
    print("Testing utility functions...")

    # Test metrics
    pred = torch.randint(0, 19, (2, 64, 64))
    target = torch.randint(0, 19, (2, 64, 64))

    f_score = calculate_f_score(pred, target)
    pixel_acc = pixel_accuracy(pred, target)

    print(f"✓ F-Score: {f_score:.4f}")
    print(f"✓ Pixel Accuracy: {pixel_acc:.4f}")

    # Test loss
    criterion = CombinedLoss()
    logits = torch.randn(2, 19, 64, 64)
    targets = torch.randint(0, 19, (2, 64, 64))

    loss = criterion(logits, targets)
    print(f"✓ Combined Loss: {loss:.4f}")

    # Test average meter
    meter = AverageMeter()
    for i in range(5):
        meter.update(i * 0.1)
    print(f"✓ Average Meter: {meter.avg:.4f}")

    print("✓ Utils test passed!")

if __name__ == "__main__":
    test_utils()