"""
Configuration file for CE7454 Face Parsing Project
"""
import os

class Config:
    # Project paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
    OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'outputs')
    CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT, 'checkpoints')

    # Dataset configuration
    NUM_CLASSES = 19  # CelebAMask-HQ has 19 classes
    IMAGE_SIZE = 512  # 512x512 as specified in project
    TRAIN_SIZE = 1000  # Training pairs
    VAL_SIZE = 100    # Validation pairs

    # Training configuration
    BATCH_SIZE = 8    # Adjustable based on GPU memory
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 100
    WEIGHT_DECAY = 1e-4

    # Model configuration
    MAX_PARAMS = 1821085  # Parameter limit from project specification

    # Data augmentation
    HORIZONTAL_FLIP_PROB = 0.5
    ROTATION_DEGREES = 15
    COLOR_JITTER_BRIGHTNESS = 0.2
    COLOR_JITTER_CONTRAST = 0.2

    # Normalization (ImageNet stats)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # Evaluation
    SAVE_BEST_ONLY = True
    PATIENCE = 10  # Early stopping patience

    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        os.makedirs(cls.DATA_ROOT, exist_ok=True)
        os.makedirs(cls.OUTPUT_ROOT, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)