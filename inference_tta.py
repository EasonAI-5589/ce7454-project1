#!/usr/bin/env python3
"""
Test-Time Augmentation (TTA) Inference for Face Parsing
Implements multi-scale + flip augmentation to improve test performance
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add models path
sys.path.insert(0, 'src/models')
from microsegformer import MicroSegFormer

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Color map for 19 face parsing classes
COLORMAP = np.array([
    [0, 0, 0],         # 0: background
    [204, 0, 0],       # 1: skin
    [76, 153, 0],      # 2: nose
    [204, 204, 0],     # 3: eye_g (glasses)
    [51, 51, 255],     # 4: l_eye
    [204, 0, 204],     # 5: r_eye
    [0, 255, 255],     # 6: l_brow
    [255, 204, 204],   # 7: r_brow
    [102, 51, 0],      # 8: l_ear
    [255, 0, 0],       # 9: r_ear
    [102, 204, 0],     # 10: mouth
    [255, 255, 0],     # 11: u_lip
    [0, 0, 153],       # 12: l_lip
    [0, 0, 204],       # 13: hair
    [255, 51, 153],    # 14: hat
    [0, 204, 204],     # 15: ear_r (earring)
    [0, 51, 0],        # 16: neck_l (necklace)
    [255, 153, 51],    # 17: neck
    [0, 204, 0]        # 18: cloth
], dtype=np.uint8)


def load_model(checkpoint_path, dropout=0.15, use_lmsa=True):
    """Load model with checkpoint"""
    model = MicroSegFormer(
        num_classes=19,
        dropout=dropout,
        use_lmsa=use_lmsa
    ).to(DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def predict_single_scale(model, img_tensor, scale=1.0):
    """
    Predict at a single scale

    Args:
        model: The segmentation model
        img_tensor: Input tensor [1, 3, H, W]
        scale: Scale factor for the image

    Returns:
        Prediction probabilities [1, 19, H, W]
    """
    if scale != 1.0:
        # Resize to scale
        orig_h, orig_w = img_tensor.shape[2:]
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        img_scaled = F.interpolate(img_tensor, size=(new_h, new_w),
                                   mode='bilinear', align_corners=False)
    else:
        img_scaled = img_tensor
        orig_h, orig_w = img_tensor.shape[2:]

    # Inference
    with torch.no_grad():
        output = model(img_scaled)  # [1, 19, H', W']

        # Apply softmax to get probabilities
        prob = F.softmax(output, dim=1)

    # Resize back to original size
    if scale != 1.0:
        prob = F.interpolate(prob, size=(orig_h, orig_w),
                            mode='bilinear', align_corners=False)

    return prob


def predict_with_tta(model, image_path, scales=[0.9, 1.0, 1.1], use_flip=True):
    """
    Predict with Test-Time Augmentation

    Args:
        model: The segmentation model
        image_path: Path to input image
        scales: List of scales to test
        use_flip: Whether to use horizontal flip

    Returns:
        Final prediction mask (H, W)
    """
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = img.shape[:2]  # (H, W)

    # Resize to 512x512 (model's training size)
    img_resized = cv2.resize(img, (512, 512))

    # Normalize
    img_normalized = img_resized.astype(np.float32) / 255.0

    # Convert to tensor
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # Accumulate predictions
    prob_sum = None
    count = 0

    for scale in scales:
        # Original orientation
        prob = predict_single_scale(model, img_tensor, scale)

        if prob_sum is None:
            prob_sum = prob
        else:
            prob_sum += prob
        count += 1

        # Horizontal flip
        if use_flip:
            img_flipped = torch.flip(img_tensor, dims=[3])  # Flip width dimension
            prob_flipped = predict_single_scale(model, img_flipped, scale)
            prob_flipped = torch.flip(prob_flipped, dims=[3])  # Flip back

            prob_sum += prob_flipped
            count += 1

    # Average predictions
    prob_avg = prob_sum / count

    # Get final prediction
    pred = torch.argmax(prob_avg, dim=1).squeeze(0).cpu().numpy()  # (512, 512)

    # Resize back to original size if needed
    if original_size != (512, 512):
        pred_resized = cv2.resize(pred.astype(np.uint8),
                                 (original_size[1], original_size[0]),
                                 interpolation=cv2.INTER_NEAREST)
    else:
        pred_resized = pred

    return pred_resized.astype(np.uint8)


def save_mask_with_palette(mask, output_path):
    """
    Save mask as PNG with palette (Codabench format)

    Args:
        mask: Prediction mask (H, W) with class indices
        output_path: Output file path
    """
    # Create palette (exact match with Codabench format)
    PALETTE = np.array([[i, i, i] for i in range(256)], dtype=np.uint8)
    PALETTE[:16] = np.array([
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [191, 0, 0],
        [64, 128, 0],
        [191, 128, 0],
        [64, 0, 128],
        [191, 0, 128],
        [64, 128, 128],
        [191, 128, 128],
    ], dtype=np.uint8)

    # Create PIL Image in palette mode
    mask_img = Image.fromarray(mask, mode='P')
    mask_img.putpalette(PALETTE.flatten().tolist())
    mask_img.save(output_path)


def main():
    parser = argparse.ArgumentParser(description='Face Parsing with TTA')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input images directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output masks directory')
    parser.add_argument('--scales', type=float, nargs='+',
                       default=[0.9, 1.0, 1.1],
                       help='Scales for multi-scale testing')
    parser.add_argument('--no-flip', action='store_true',
                       help='Disable horizontal flip augmentation')
    parser.add_argument('--dropout', type=float, default=0.15,
                       help='Model dropout rate')
    parser.add_argument('--use-lmsa', action='store_true', default=True,
                       help='Use LMSA module')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Face Parsing with Test-Time Augmentation (TTA)")
    print("=" * 80)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Input dir: {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device: {DEVICE}")
    print(f"\nTTA Settings:")
    print(f"  Scales: {args.scales}")
    print(f"  Horizontal flip: {not args.no_flip}")
    print(f"  Total augmentations: {len(args.scales) * (2 if not args.no_flip else 1)}")

    # Load model
    print(f"\nLoading model...")
    model = load_model(args.checkpoint, dropout=args.dropout, use_lmsa=args.use_lmsa)
    print(f"✅ Model loaded successfully!")
    print(f"   LMSA module: {'ENABLED' if args.use_lmsa else 'DISABLED'}")

    # Get all image files
    image_files = sorted([f for f in os.listdir(args.input_dir)
                         if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    print(f"\n📁 Found {len(image_files)} images")
    print(f"\nProcessing with TTA...\n")

    # Process each image
    for idx, image_file in enumerate(tqdm(image_files, desc="Generating masks"), 1):
        image_path = os.path.join(args.input_dir, image_file)

        # Generate mask with TTA
        mask = predict_with_tta(model, image_path,
                               scales=args.scales,
                               use_flip=not args.no_flip)

        # Save mask
        output_filename = os.path.splitext(image_file)[0] + '.png'
        output_path = os.path.join(args.output_dir, output_filename)
        save_mask_with_palette(mask, output_path)

    print(f"\n" + "=" * 80)
    print(f"✅ Complete! {len(image_files)} masks generated with TTA")
    print(f"📁 Output: {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
