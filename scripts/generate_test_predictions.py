#!/usr/bin/env python3
"""
Generate test predictions for Codabench submission
Using the best model checkpoint
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.microsegformer import MicroSegFormer

def generate_predictions(checkpoint_path, test_dir, output_dir):
    """Generate predictions for all test images"""

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = MicroSegFormer(num_classes=19, dropout=0.15, use_lmsa=True)
    model = model.to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from: {checkpoint_path}")

    # Get test images
    test_images = sorted(Path(test_dir).glob('*.jpg'))
    print(f"Found {len(test_images)} test images")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate predictions
    with torch.no_grad():
        for img_path in tqdm(test_images, desc="Generating predictions"):
            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')
            img = img.resize((512, 512))
            img_array = np.array(img).astype(np.float32) / 255.0

            # Convert to tensor
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)

            # Predict
            output = model(img_tensor)
            mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # Save mask with palette
            PALETTE = np.array([[i, i, i] for i in range(256)])
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
            ])

            mask_img = Image.fromarray(mask)
            mask_img.putpalette(PALETTE.reshape(-1).tolist())

            # Save with same name as input (but .png)
            output_path = Path(output_dir) / f"{img_path.stem}.png"
            mask_img.save(output_path)

    print(f"\n✅ Generated {len(test_images)} predictions in: {output_dir}")

if __name__ == "__main__":
    # Configuration
    CHECKPOINT = "checkpoints/microsegformer_20251008_025917/best_model.pth"
    TEST_DIR = "data/test_public/images"
    OUTPUT_DIR = "submission/masks_new"

    print("="*80)
    print("GENERATING TEST PREDICTIONS")
    print("="*80)
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Test images: {TEST_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*80)

    generate_predictions(CHECKPOINT, TEST_DIR, OUTPUT_DIR)
