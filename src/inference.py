"""
Inference script for generating test set predictions
"""
import torch
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

from config import Config
from dataset import CelebAMaskDataset, get_transforms
from models.unet import get_model
from utils import load_checkpoint

def predict_single_image(model, image_path, device, output_path=None):
    """Predict single image"""
    model.eval()

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = get_transforms('test')

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Convert to PIL image and save
    pred_image = Image.fromarray(pred.astype(np.uint8))

    if output_path:
        pred_image.save(output_path)

    return pred_image

def predict_test_set(model, test_data_root, output_dir, device):
    """Generate predictions for entire test set"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get test image paths
    test_image_dir = os.path.join(test_data_root, 'test', 'images')

    if not os.path.exists(test_image_dir):
        print(f"Test image directory not found: {test_image_dir}")
        return

    image_files = [f for f in os.listdir(test_image_dir)
                   if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()

    print(f"Found {len(image_files)} test images")

    model.eval()

    for img_file in tqdm(image_files, desc="Generating predictions"):
        img_path = os.path.join(test_image_dir, img_file)

        # Output filename (same as input but .png)
        output_filename = os.path.splitext(img_file)[0] + '.png'
        output_path = os.path.join(output_dir, output_filename)

        # Generate prediction
        predict_single_image(model, img_path, device, output_path)

    print(f"Predictions saved to: {output_dir}")

def batch_predict_test_set(model, test_data_root, output_dir, device, batch_size=8):
    """Generate predictions using batch processing"""

    os.makedirs(output_dir, exist_ok=True)

    # Create dataset and dataloader for test set
    transform = get_transforms('test')
    test_dataset = CelebAMaskDataset(
        data_root=test_data_root,
        split='test',
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    model.eval()

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(test_loader, desc="Batch prediction")):
            images = images.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

            # Save each prediction
            for i, pred in enumerate(predictions):
                img_idx = batch_idx * batch_size + i
                if img_idx < len(test_dataset.image_files):
                    img_filename = test_dataset.image_files[img_idx]
                    output_filename = os.path.splitext(img_filename)[0] + '.png'
                    output_path = os.path.join(output_dir, output_filename)

                    pred_image = Image.fromarray(pred.astype(np.uint8))
                    pred_image.save(output_path)

    print(f"Batch predictions saved to: {output_dir}")

def test_time_augmentation(model, image_path, device, num_augs=5):
    """Apply test-time augmentation for better results"""
    from torchvision import transforms

    model.eval()
    image = Image.open(image_path).convert('RGB')

    # Define augmentations
    base_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.MEAN, std=Config.STD)
    ])

    aug_transforms = [
        base_transform,  # Original
        transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN, std=Config.STD)
        ]),
        transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN, std=Config.STD)
        ]),
        transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.RandomRotation(degrees=-5),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN, std=Config.STD)
        ]),
        transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN, std=Config.STD)
        ])
    ]

    predictions = []

    with torch.no_grad():
        for transform in aug_transforms[:num_augs]:
            image_tensor = transform(image).unsqueeze(0).to(device)
            output = model(image_tensor)
            pred = torch.softmax(output, dim=1).squeeze().cpu().numpy()
            predictions.append(pred)

    # Average predictions
    avg_pred = np.mean(predictions, axis=0)
    final_pred = np.argmax(avg_pred, axis=0)

    return final_pred

def main():
    """Main inference function"""
    import argparse

    parser = argparse.ArgumentParser(description='Face Parsing Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, default=Config.DATA_ROOT, help='Test data root directory')
    parser.add_argument('--output_dir', type=str, default='outputs/test_predictions', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--model_name', type=str, default='resnet_unet', help='Model architecture')
    parser.add_argument('--tta', action='store_true', help='Use test-time augmentation')

    args = parser.parse_args()

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}")
    model = get_model(args.model_name, n_classes=Config.NUM_CLASSES)
    model = model.to(device)

    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully!")

    # Generate predictions
    if args.tta:
        print("Using test-time augmentation (single image processing)")
        predict_test_set(model, args.test_data, args.output_dir, device)
    else:
        print("Using batch processing")
        batch_predict_test_set(model, args.test_data, args.output_dir, device, args.batch_size)

    print("Inference completed!")

if __name__ == "__main__":
    main()