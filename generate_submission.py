#!/usr/bin/env python3
"""
Generate Codabench submission
Usage: python generate_submission.py --model checkpoints/xxx/best_model.pth
"""
import torch
import os
import argparse
from PIL import Image
import numpy as np
import zipfile
from tqdm import tqdm

from src.models.microsegformer import MicroSegFormer

def predict_image(model, image_path, device):
    """Predict single image"""
    model.eval()

    # Load and preprocess
    image = Image.open(image_path).convert('RGB')
    image = image.resize((512, 512))

    # To tensor
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    return pred.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--test-dir', default='data/test/images', help='Test images directory')
    parser.add_argument('--output', default='submission', help='Output directory')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"Loading model: {args.model}")
    model = MicroSegFormer(num_classes=19).to(device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded from epoch {checkpoint.get('epoch', 0)}")

    # Create output structure
    masks_dir = os.path.join(args.output, 'masks')
    solution_dir = os.path.join(args.output, 'solution')
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(solution_dir, exist_ok=True)

    # Get test images
    test_images = sorted([f for f in os.listdir(args.test_dir) if f.endswith('.jpg')])
    print(f"Found {len(test_images)} test images")

    # Generate predictions
    print("Generating predictions...")
    for img_file in tqdm(test_images):
        img_path = os.path.join(args.test_dir, img_file)
        pred = predict_image(model, img_path, device)

        # Save as PNG (single channel)
        output_file = img_file.replace('.jpg', '.png')
        output_path = os.path.join(masks_dir, output_file)
        Image.fromarray(pred, mode='L').save(output_path)

    # Create submission zip
    zip_path = f"{args.output}.zip"
    print(f"Creating {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # Add solution folder (empty)
        zipf.writestr('solution/', '')

        # Add masks
        for mask_file in os.listdir(masks_dir):
            file_path = os.path.join(masks_dir, mask_file)
            arcname = os.path.join('masks', mask_file)
            zipf.write(file_path, arcname)

    print(f"\n✅ Submission ready: {zip_path}")
    print(f"   - {len(test_images)} masks in masks/")
    print(f"   - solution/ folder (empty)")
    print("\nNext steps:")
    print("1. Upload to Codabench")
    print("2. Wait for results")

if __name__ == '__main__':
    main()
