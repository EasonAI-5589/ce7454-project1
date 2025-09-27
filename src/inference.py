"""
Inference script for generating test predictions
Codabench submission format
"""
import torch
import os
import argparse
from PIL import Image
import numpy as np

from model import FaceParsingNet
from utils import load_checkpoint

def predict_single_image(model, image_path, device, image_size=512):
    """Predict single image"""
    model.eval()

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((image_size, image_size))

    # Convert to tensor
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    return pred.astype(np.uint8)

def generate_test_predictions(model_path, data_root='data', output_dir='predictions'):
    """Generate predictions for all test images"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {model_path}")
    model = FaceParsingNet(n_classes=19)
    model = model.to(device)

    # Load checkpoint
    try:
        epoch, loss = load_checkpoint(model, None, model_path, device)
        print(f"Loaded model from epoch {epoch}, loss: {loss:.4f}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get test images
    test_image_dir = os.path.join(data_root, 'test', 'images')
    if not os.path.exists(test_image_dir):
        print(f"Test image directory not found: {test_image_dir}")
        return

    image_files = [f for f in os.listdir(test_image_dir)
                   if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()

    print(f"Found {len(image_files)} test images")

    # Generate predictions
    for i, img_file in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {img_file}")

        img_path = os.path.join(test_image_dir, img_file)
        pred = predict_single_image(model, img_path, device)

        # Save prediction with same filename but .png extension
        output_filename = os.path.splitext(img_file)[0] + '.png'
        output_path = os.path.join(output_dir, output_filename)

        pred_image = Image.fromarray(pred, mode='L')  # Single channel
        pred_image.save(output_path)

    print(f"Predictions saved to: {output_dir}")

def create_submission_zip(predictions_dir, output_zip='submission.zip'):
    """Create Codabench submission zip"""
    import zipfile

    with zipfile.ZipFile(output_zip, 'w') as zipf:
        # Add solution folder (empty as required)
        zipf.writestr('solution/', '')

        # Add masks
        masks_dir = 'masks'
        for filename in os.listdir(predictions_dir):
            if filename.endswith('.png'):
                file_path = os.path.join(predictions_dir, filename)
                arcname = os.path.join(masks_dir, filename)
                zipf.write(file_path, arcname)

    print(f"Submission zip created: {output_zip}")

def main():
    parser = argparse.ArgumentParser(description='Generate test predictions')
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--data', default='data', help='Data root directory')
    parser.add_argument('--output', default='predictions', help='Output directory')
    parser.add_argument('--zip', action='store_true', help='Create submission zip')

    args = parser.parse_args()

    print("="*60)
    print("Face Parsing Inference")
    print("="*60)

    # Generate predictions
    generate_test_predictions(args.model, args.data, args.output)

    # Create submission zip if requested
    if args.zip:
        create_submission_zip(args.output)

    print("\nInference completed!")

if __name__ == "__main__":
    main()