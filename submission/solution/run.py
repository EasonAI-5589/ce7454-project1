# -*- coding: utf-8 -*-
#
# CE7454 Face Parsing - Codabench Submission
# MicroSegFormer Inference Script
#

import argparse
import sys
import os
import cv2
import torch
import numpy as np
from PIL import Image

# Add current directory to path to import model
sys.path.insert(0, os.path.dirname(__file__))

from microsegformer import MicroSegFormer


def main(input, output, weights):
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the input image
    img = cv2.imread(input)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to 512x512
    img = cv2.resize(img, (512, 512))

    # Normalize: convert to float and scale to [0, 1]
    img = img.astype(np.float32) / 255.0

    # Convert to PyTorch tensor and add batch dimension
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    # Initialize the model (with dropout parameter for compatibility)
    model = MicroSegFormer(num_classes=19, dropout=0.15)
    model = model.to(device)

    # Load the checkpoint
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    # Load state_dict from checkpoint
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Inference
    with torch.no_grad():
        prediction = model(img_tensor)
        # Get class predictions
        mask = torch.argmax(prediction, dim=1).squeeze(0).cpu().numpy()

    # Convert to uint8 for saving
    mask = mask.astype(np.uint8)

    # Save as single-channel PNG
    Image.fromarray(mask).save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--weights", type=str, default="ckpt.pth")
    args = parser.parse_args()
    main(args.input, args.output, args.weights)
