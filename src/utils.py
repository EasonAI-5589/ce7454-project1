"""
Simple utilities for face parsing
Only depends on PyTorch and numpy
"""
import torch
import torch.nn as nn
import numpy as np
import os

def calculate_f_score(pred, target, beta=1):
    """
    Calculate F-Score for segmentation
    EXACT match with Codabench evaluation code
    Reference: https://www.codabench.org/competitions/2681/

    Note: Number of classes is automatically determined from unique values in target (mask_gt)
    """
    # Convert to numpy if tensor
    if hasattr(pred, 'cpu'):
        pred = pred.cpu().numpy()
    if hasattr(target, 'cpu'):
        target = target.cpu().numpy()

    # Codabench uses mask_gt, mask_pred naming
    mask_gt = target
    mask_pred = pred

    f_scores = []

    # EXACT Codabench implementation: iterate over unique classes in ground truth
    # No need for num_classes - dynamically get from np.unique(mask_gt)
    for class_id in np.unique(mask_gt):
        tp = np.sum((mask_gt == class_id) & (mask_pred == class_id))
        fp = np.sum((mask_gt != class_id) & (mask_pred == class_id))
        fn = np.sum((mask_gt == class_id) & (mask_pred != class_id))

        # EXACT Codabench formula
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f_score = (
            (1 + beta**2)
            * (precision * recall)
            / ((beta**2 * precision) + recall + 1e-7)
        )

        f_scores.append(f_score)

    return np.mean(f_scores)

def pixel_accuracy(pred, target):
    """Calculate pixel accuracy"""
    correct = (pred == target).sum().float()
    total = target.numel()
    return (correct / total).item()

class CombinedLoss(nn.Module):
    """Combined Cross Entropy + Dice Loss with Class Weights"""

    def __init__(self, ce_weight=1.0, dice_weight=0.5, class_weights=None, use_focal=False, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # Initialize CE loss with class weights
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

        print(f"Loss initialized: CE_weight={ce_weight}, Dice_weight={dice_weight}, "
              f"Class_weights={'enabled' if class_weights is not None else 'disabled'}, "
              f"Focal={'enabled' if use_focal else 'disabled'}")

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

    def focal_loss(self, inputs, targets):
        """
        Focal Loss for handling hard examples
        FL(p_t) = -α(1-p_t)^γ * log(p_t)
        """
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - p_t) ** self.focal_gamma * ce_loss
        return focal_loss.mean()

    def forward(self, inputs, targets):
        # Choose between CE and Focal Loss
        if self.use_focal:
            ce = self.focal_loss(inputs, targets)
        else:
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