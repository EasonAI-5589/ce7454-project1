#!/bin/bash
# CE7454 Project 1 - Environment Setup Script for CUDA Server

echo "================================================"
echo "CE7454 Face Parsing - Environment Setup"
echo "================================================"

# 1. Create conda environment
echo -e "\n[1/5] Creating conda environment..."
conda create -n ce7454 python=3.9 -y

# 2. Activate environment
echo -e "\n[2/5] Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ce7454

# 3. Install PyTorch with CUDA
echo -e "\n[3/5] Installing PyTorch with CUDA 11.8..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 4. Install other dependencies
echo -e "\n[4/5] Installing other dependencies..."
pip install numpy pillow matplotlib

# 5. Verify installation
echo -e "\n[5/5] Verifying installation..."
python << PYEOF
import torch
import numpy as np
from PIL import Image

print("✓ PyTorch version:", torch.__version__)
print("✓ CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("✓ CUDA version:", torch.version.cuda)
    print("✓ GPU device:", torch.cuda.get_device_name(0))
print("✓ NumPy version:", np.__version__)
print("✓ Pillow version:", Image.__version__)
PYEOF

echo -e "\n================================================"
echo "Environment setup completed!"
echo "Run: conda activate ce7454"
echo "================================================"
