"""
MicroSegFormer - Lightweight Transformer for Face Parsing
Inspired by SegFormer with parameter optimization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OverlapPatchEmbed(nn.Module):
    """Overlapping Patch Embedding with convolution"""
    def __init__(self, patch_size=7, stride=4, in_channels=3, embed_dim=32):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # B, C, H, W
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        x = self.norm(x)
        return x, H, W


class EfficientSelfAttention(nn.Module):
    """Efficient Self-Attention with reduced KV sequence length and dropout"""
    def __init__(self, dim, num_heads=1, sr_ratio=1, attn_dropout=0.1, proj_dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

        # Dropout layers
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        # Spatial reduction for efficient attention
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape

        # Query
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Key and Value with spatial reduction
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)  # Apply attention dropout

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)  # Apply projection dropout

        return x


class MLP(nn.Module):
    """Feed-forward network with dropout"""
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Block with Efficient Attention and Dropout"""
    def __init__(self, dim, num_heads=1, mlp_ratio=2, sr_ratio=1, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio,
                                           attn_dropout=dropout, proj_dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_features=int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x


class LMSA(nn.Module):
    """
    Lightweight Multi-Scale Attention Module
    Designed for fine-grained face parsing with small objects (eyes, ears)

    Key features:
    - Depthwise separable convolutions for efficiency
    - Channel-wise attention for adaptive feature weighting
    - Lightweight design (~20k params total)
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        # Depthwise separable multi-scale paths (much more efficient)
        self.dw_conv3x3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

        # SE-style channel attention (more aggressive reduction)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Depthwise separable convolution
        out = self.dw_conv3x3(x)
        out = self.pw_conv(out)
        out = self.bn(out)

        # Channel attention
        gap = self.global_pool(out).squeeze(-1).squeeze(-1)
        attention = self.fc(gap).unsqueeze(-1).unsqueeze(-1)

        # Apply attention and add residual
        return out * attention + x


class MLPDecoder(nn.Module):
    """Lightweight MLP decoder with LMSA for multi-scale feature fusion"""
    def __init__(self, in_channels=[32, 64, 128, 256], embed_dim=128, num_classes=19, use_lmsa=True):
        super().__init__()
        self.use_lmsa = use_lmsa

        # Linear layers to unify channel dimensions
        self.linear_c4 = nn.Linear(in_channels[3], embed_dim)
        self.linear_c3 = nn.Linear(in_channels[2], embed_dim)
        self.linear_c2 = nn.Linear(in_channels[1], embed_dim)
        self.linear_c1 = nn.Linear(in_channels[0], embed_dim)

        # LMSA modules for each decoder stage
        if use_lmsa:
            self.lmsa = LMSA(embed_dim, reduction=4)
            print(f"✓ LMSA module enabled in decoder")

        # Fusion MLP
        self.linear_fuse = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Final prediction head
        self.conv_seg = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, features):
        c1, c2, c3, c4 = features
        B = c1.shape[0]

        # Get spatial dimensions
        _, _, H1, W1 = c1.shape

        # Unify channels: B, N, C -> B, C, H, W
        _c4 = self.linear_c4(c4.flatten(2).transpose(1, 2))
        _c3 = self.linear_c3(c3.flatten(2).transpose(1, 2))
        _c2 = self.linear_c2(c2.flatten(2).transpose(1, 2))
        _c1 = self.linear_c1(c1.flatten(2).transpose(1, 2))

        # Upsample to c1 size
        _c4 = _c4.permute(0, 2, 1).reshape(B, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=(H1, W1), mode='bilinear', align_corners=False)
        _c4 = _c4.flatten(2).transpose(1, 2)

        _c3 = _c3.permute(0, 2, 1).reshape(B, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=(H1, W1), mode='bilinear', align_corners=False)
        _c3 = _c3.flatten(2).transpose(1, 2)

        _c2 = _c2.permute(0, 2, 1).reshape(B, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=(H1, W1), mode='bilinear', align_corners=False)
        _c2 = _c2.flatten(2).transpose(1, 2)

        # Concatenate and fuse
        _c = torch.cat([_c1, _c2, _c3, _c4], dim=-1)
        _c = self.linear_fuse(_c)

        # Reshape to image
        _c = _c.permute(0, 2, 1).reshape(B, -1, H1, W1)

        # Apply LMSA for multi-scale attention
        if self.use_lmsa:
            _c = self.lmsa(_c)

        # Upsample
        _c = F.interpolate(_c, scale_factor=4, mode='bilinear', align_corners=False)

        # Final prediction
        x = self.conv_seg(_c)
        return x


class MicroSegFormer(nn.Module):
    """Ultra-lightweight SegFormer for face parsing (<1.82M params) with dropout regularization and LMSA"""
    def __init__(self, num_classes=19, embed_dims=None, depths=None, sr_ratios=None, dropout=0.15, use_lmsa=True):
        super().__init__()

        # Default configuration - optimized for ~1.7M params
        if embed_dims is None:
            embed_dims = [32, 64, 128, 192]
        if depths is None:
            depths = [1, 2, 2, 2]
        if sr_ratios is None:
            sr_ratios = [8, 4, 2, 1]

        self.embed_dims = embed_dims
        self.depths = depths

        num_heads = [1, 1, 1, 1]
        mlp_ratios = [2, 2, 2, 2]

        # Patch embeddings for each stage
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_channels=3, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_channels=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_channels=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_channels=embed_dims[2], embed_dim=embed_dims[3])

        # Transformer blocks with dropout
        self.block1 = nn.ModuleList([
            TransformerBlock(embed_dims[0], num_heads[0], mlp_ratios[0], sr_ratios[0], dropout=dropout*0.5)
            for _ in range(depths[0])
        ])
        self.block2 = nn.ModuleList([
            TransformerBlock(embed_dims[1], num_heads[1], mlp_ratios[1], sr_ratios[1], dropout=dropout*0.75)
            for _ in range(depths[1])
        ])
        self.block3 = nn.ModuleList([
            TransformerBlock(embed_dims[2], num_heads[2], mlp_ratios[2], sr_ratios[2], dropout=dropout)
            for _ in range(depths[2])
        ])
        self.block4 = nn.ModuleList([
            TransformerBlock(embed_dims[3], num_heads[3], mlp_ratios[3], sr_ratios[3], dropout=dropout*1.25)
            for _ in range(depths[3])
        ])

        # Norms
        self.norm1 = nn.LayerNorm(embed_dims[0])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        self.norm4 = nn.LayerNorm(embed_dims[3])

        # Decoder with LMSA
        self.decoder = MLPDecoder(in_channels=embed_dims, embed_dim=128, num_classes=num_classes, use_lmsa=use_lmsa)

    def forward(self, x):
        B = x.shape[0]

        # Stage 1
        x, H1, W1 = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H1, W1)
        x1 = self.norm1(x).reshape(B, H1, W1, -1).permute(0, 3, 1, 2)

        # Stage 2
        x, H2, W2 = self.patch_embed2(x1)
        for blk in self.block2:
            x = blk(x, H2, W2)
        x2 = self.norm2(x).reshape(B, H2, W2, -1).permute(0, 3, 1, 2)

        # Stage 3
        x, H3, W3 = self.patch_embed3(x2)
        for blk in self.block3:
            x = blk(x, H3, W3)
        x3 = self.norm3(x).reshape(B, H3, W3, -1).permute(0, 3, 1, 2)

        # Stage 4
        x, H4, W4 = self.patch_embed4(x3)
        for blk in self.block4:
            x = blk(x, H4, W4)
        x4 = self.norm4(x).reshape(B, H4, W4, -1).permute(0, 3, 1, 2)

        # Decode
        out = self.decoder([x1, x2, x3, x4])

        return out


def test_microsegformer():
    """Test MicroSegFormer"""
    model = MicroSegFormer(num_classes=19)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"MicroSegFormer Test")
    print(f"Parameters: {param_count:,}")
    print(f"Within limit: {param_count < 1821085}")

    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        y = model(x)

    print(f"Input: {x.shape} -> Output: {y.shape}")
    return model


if __name__ == "__main__":
    test_microsegformer()
