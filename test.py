#!/usr/bin/env python3
"""
Test script for evaluating trained model
"""

import argparse
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
from src.dataset import create_train_val_loaders
from src.models.microsegformer import MicroSegFormer


def compute_metrics(pred, target, num_classes=19):
    """Compute F-Score metrics"""
    pred = pred.argmax(dim=1)  # B x H x W

    f_scores = []
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)

        intersection = (pred_mask & target_mask).sum().float()
        pred_sum = pred_mask.sum().float()
        target_sum = target_mask.sum().float()

        if pred_sum + target_sum == 0:
            f_scores.append(1.0)  # Perfect score if class not present
        else:
            precision = intersection / (pred_sum + 1e-10)
            recall = intersection / (target_sum + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            f_scores.append(f1.item())

    return sum(f_scores) / len(f_scores)


def test_model(checkpoint_path, device='cuda'):
    """Test model on validation set"""

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint or use default
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Default config
        with open('configs/main.yaml', 'r') as f:
            config = yaml.safe_load(f)

    # Setup model
    model = MicroSegFormer(num_classes=config['model']['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from: {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'best_f_score' in checkpoint:
        print(f"Best F-Score: {checkpoint['best_f_score']:.4f}")

    # Setup dataset - use validation split
    print("\nLoading validation dataset...")
    _, val_loader = create_train_val_loaders(
        data_root=config['data']['root'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        val_split=config['data']['val_split']
    )

    print(f"Validation set: {len(val_loader.dataset)} samples")

    # Test
    total_f_score = 0
    num_batches = 0

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc='Testing'):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            f_score = compute_metrics(outputs, masks, config['model']['num_classes'])
            total_f_score += f_score
            num_batches += 1

    avg_f_score = total_f_score / num_batches
    print(f"\nValidation F-Score: {avg_f_score:.4f}")

    return avg_f_score


def main():
    parser = argparse.ArgumentParser(description='Test face parsing model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    test_model(args.checkpoint, device)


if __name__ == '__main__':
    main()
