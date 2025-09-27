"""
U-Net implementation for face parsing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
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

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=19, bilinear=True):
        super(UNet, self).__init__()
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

class ResNetUNet(nn.Module):
    """U-Net with ResNet backbone"""

    def __init__(self, n_classes=19, backbone='resnet34'):
        super(ResNetUNet, self).__init__()
        self.n_classes = n_classes

        # Load pretrained ResNet (but we'll train from scratch as required)
        if backbone == 'resnet34':
            resnet = models.resnet34(pretrained=False)  # No pretrained weights
            self.encoder_channels = [64, 64, 128, 256, 512]
        elif backbone == 'resnet18':
            resnet = models.resnet18(pretrained=False)
            self.encoder_channels = [64, 64, 128, 256, 512]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Extract encoder layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Decoder
        self.up1 = Up(512 + 256, 256, bilinear=True)
        self.up2 = Up(256 + 128, 128, bilinear=True)
        self.up3 = Up(128 + 64, 64, bilinear=True)
        self.up4 = Up(64 + 64, 64, bilinear=True)

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.relu(self.bn1(self.conv1(x)))  # 64
        x1_pool = self.maxpool(x1)

        x2 = self.layer1(x1_pool)  # 64
        x3 = self.layer2(x2)       # 128
        x4 = self.layer3(x3)       # 256
        x5 = self.layer4(x4)       # 512

        # Decoder
        x = self.up1(x5, x4)  # 256
        x = self.up2(x, x3)   # 128
        x = self.up3(x, x2)   # 64
        x = self.up4(x, x1)   # 64

        x = self.final_conv(x)
        return x

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model(model_name='unet', n_classes=19):
    """Get model by name"""
    if model_name == 'unet':
        model = UNet(n_channels=3, n_classes=n_classes)
    elif model_name == 'resnet_unet':
        model = ResNetUNet(n_classes=n_classes, backbone='resnet34')
    elif model_name == 'resnet18_unet':
        model = ResNetUNet(n_classes=n_classes, backbone='resnet18')
    else:
        raise ValueError(f"Unknown model: {model_name}")

    param_count = count_parameters(model)
    print(f"Model: {model_name}")
    print(f"Parameters: {param_count:,}")
    print(f"Within limit: {param_count < 1821085}")

    return model

if __name__ == "__main__":
    # Test models
    models_to_test = ['unet', 'resnet_unet', 'resnet18_unet']

    for model_name in models_to_test:
        print(f"\n--- Testing {model_name} ---")
        model = get_model(model_name)

        # Test forward pass
        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            y = model(x)
        print(f"Output shape: {y.shape}")
        print(f"Expected: (1, 19, 512, 512)")
        print("✓ Forward pass successful" if y.shape == (1, 19, 512, 512) else "✗ Forward pass failed")