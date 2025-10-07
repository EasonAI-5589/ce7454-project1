"""
U-Net - Classic CNN Architecture for Semantic Segmentation
Original paper: https://arxiv.org/abs/1505.04597
Optimized for parameter efficiency (<1.82M params)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block (Conv -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, dropout=0.0):
        super().__init__()

        # Use bilinear upsampling (no parameters)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout=dropout)
        else:
            # Transposed convolution (has parameters)
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle size mismatch (input image may not be perfectly divisible)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concatenate along channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for Face Parsing
    Optimized for parameter efficiency

    Args:
        n_channels: Number of input channels (3 for RGB)
        n_classes: Number of output classes (19 for face parsing)
        base_channels: Base number of channels (default: 32)
        bilinear: Use bilinear upsampling (True) or transposed conv (False)
        dropout: Dropout rate (0.0 to disable)
    """
    def __init__(self, n_channels=3, n_classes=19, base_channels=32, bilinear=True, dropout=0.0):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dropout = dropout

        # Encoder
        self.inc = DoubleConv(n_channels, base_channels, dropout=dropout)
        self.down1 = Down(base_channels, base_channels * 2, dropout=dropout)
        self.down2 = Down(base_channels * 2, base_channels * 4, dropout=dropout)
        self.down3 = Down(base_channels * 4, base_channels * 8, dropout=dropout)

        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor, dropout=dropout)

        # Decoder
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear, dropout=dropout)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear, dropout=dropout)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear, dropout=dropout)
        self.up4 = Up(base_channels * 2, base_channels, bilinear, dropout=dropout)

        # Output layer
        self.outc = nn.Conv2d(base_channels, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)      # base_channels
        x2 = self.down1(x1)   # base_channels * 2
        x3 = self.down2(x2)   # base_channels * 4
        x4 = self.down3(x3)   # base_channels * 8
        x5 = self.down4(x4)   # base_channels * 16 (bottleneck)

        # Decoder with skip connections
        x = self.up1(x5, x4)  # base_channels * 8
        x = self.up2(x, x3)   # base_channels * 4
        x = self.up3(x, x2)   # base_channels * 2
        x = self.up4(x, x1)   # base_channels

        # Output
        logits = self.outc(x)
        return logits


def test_unet():
    """Test U-Net model"""
    print("Testing U-Net model...")

    # Test different configurations (optimized for <1.82M params)
    configs = [
        {"base_channels": 16, "bilinear": True, "dropout": 0.0},
        {"base_channels": 18, "bilinear": True, "dropout": 0.0},
        {"base_channels": 20, "bilinear": True, "dropout": 0.15},
        {"base_channels": 22, "bilinear": True, "dropout": 0.15},
    ]

    param_limit = 1821085

    for i, config in enumerate(configs):
        model = UNet(n_channels=3, n_classes=19, **config)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        usage_percent = (total_params / param_limit) * 100

        print(f"\n{'='*60}")
        print(f"Config {i+1}: {config}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Parameter usage: {usage_percent:.1f}%")
        print(f"Within limit: {'✓' if total_params <= param_limit else '✗'}")

        # Test forward pass
        x = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            y = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
        assert y.shape == (2, 19, 512, 512), f"Output shape mismatch: {y.shape}"

    print(f"\n{'='*60}")
    print("✓ U-Net test passed!")


if __name__ == "__main__":
    test_unet()
