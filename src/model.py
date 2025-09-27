"""
Lightweight U-Net for Face Parsing
Only depends on PyTorch core
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Double convolution block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle size differences
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SimpleFaceParsingNet(nn.Module):
    """Very lightweight U-Net for face parsing"""
    def __init__(self, n_classes=19):
        super().__init__()
        self.n_classes = n_classes

        # Encoder - much smaller channels
        self.inc = DoubleConv(3, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)

        # Decoder
        self.up1 = Up(128, 64, 64)   # 128 -> 64, skip: 64
        self.up2 = Up(64, 32, 32)    # 64 -> 32, skip: 32
        self.up3 = Up(32, 16, 16)    # 32 -> 16, skip: 16

        # Output
        self.outc = nn.Conv2d(16, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)       # 16
        x2 = self.down1(x1)    # 32
        x3 = self.down2(x2)    # 64
        x4 = self.down3(x3)    # 128

        x = self.up1(x4, x3)   # 128 + 64 -> 64
        x = self.up2(x, x2)    # 64 + 32 -> 32
        x = self.up3(x, x1)    # 32 + 16 -> 16

        return self.outc(x)

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model():
    """Test model creation and forward pass"""
    model = SimpleFaceParsingNet(n_classes=19)
    param_count = count_parameters(model)

    print(f"Model created successfully!")
    print(f"Parameters: {param_count:,}")
    print(f"Within limit (<1,821,085): {param_count < 1821085}")

    # Test forward pass
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        y = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Expected: (1, 19, 512, 512)")

    success = (param_count < 1821085) and (y.shape == (1, 19, 512, 512))
    print(f"Test result: {'✓ PASS' if success else '✗ FAIL'}")

    return success

# Alias for backward compatibility
FaceParsingNet = SimpleFaceParsingNet

if __name__ == "__main__":
    test_model()