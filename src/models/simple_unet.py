"""
Simple U-Net implementation without external dependencies
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
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

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle input with differing sizes
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class SimpleUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=19, bilinear=True):
        super(SimpleUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# Lightweight ResNet-like backbone blocks
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class LightweightResNetUNet(nn.Module):
    """Lightweight ResNet-inspired U-Net"""

    def __init__(self, n_classes=19):
        super(LightweightResNetUNet, self).__init__()
        self.n_classes = n_classes

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Encoder blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)     # 64 -> 64
        self.layer2 = self._make_layer(64, 128, 2, stride=2)    # 64 -> 128
        self.layer3 = self._make_layer(128, 256, 2, stride=2)   # 128 -> 256
        self.layer4 = self._make_layer(256, 512, 2, stride=2)   # 256 -> 512

        # Decoder
        self.up1 = Up(512 + 256, 256, bilinear=True)
        self.up2 = Up(256 + 128, 128, bilinear=True)
        self.up3 = Up(128 + 64, 64, bilinear=True)
        self.up4 = Up(64 + 64, 64, bilinear=True)

        # Final upsampling and classification
        self.final_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        x1 = self.relu(self.bn1(self.conv1(x)))
        x1_pool = self.maxpool(x1)

        x2 = self.layer1(x1_pool)  # 64 channels
        x3 = self.layer2(x2)       # 128 channels
        x4 = self.layer3(x3)       # 256 channels
        x5 = self.layer4(x4)       # 512 channels

        # Decoder
        x = self.up1(x5, x4)  # 256 channels
        x = self.up2(x, x3)   # 128 channels
        x = self.up3(x, x2)   # 64 channels
        x = self.up4(x, x1)   # 64 channels

        # Final upsampling and classification
        x = self.final_upsample(x)
        x = self.final_conv(x)

        return x

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_simple_model(model_name='simple_unet', n_classes=19):
    """Get model by name"""
    if model_name == 'simple_unet':
        model = SimpleUNet(n_channels=3, n_classes=n_classes)
    elif model_name == 'lightweight_resnet_unet':
        model = LightweightResNetUNet(n_classes=n_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    param_count = count_parameters(model)
    print(f"Model: {model_name}")
    print(f"Parameters: {param_count:,}")
    print(f"Within limit: {param_count < 1821085}")

    return model

if __name__ == "__main__":
    # Test models
    models_to_test = ['simple_unet', 'lightweight_resnet_unet']

    for model_name in models_to_test:
        print(f"\n--- Testing {model_name} ---")
        model = get_simple_model(model_name)

        # Test forward pass
        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            y = model(x)
        print(f"Output shape: {y.shape}")
        print(f"Expected: (1, 19, 512, 512)")
        print("✓ Forward pass successful" if y.shape == (1, 19, 512, 512) else "✗ Forward pass failed")