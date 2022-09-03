import torch
import torch.nn as nn


class MLP(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

# Residual addition skip connection

class ResAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class Embedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=14, emb_size=256, n_patches = 128):
        self.patch_size = 16
        super().__init__()

        # no need to rearrange here... flatten will do the work for us
        self.projection = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear((patch_size ** 2) * in_channels, emb_size)
        )

        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn(n_patches, emb_size))

    def forward(self, x):
        # input is [B, C, H, W]
        # Steps...
        # 1. Flatten
        # 2. Linear transformation
        x = self.projection(x)
        # 3. Patch embedding
        # Only need to do positional encoding in the case of segmentation
        # Don't need classification tokens? Maybe
        x += self.positions
        # 4. Return to transformer encoder block
        return x

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

class DownConv(nn.Sequential):
    def __init__(self, in_count, out_count):
        super().__init__(
            nn.Conv2d(in_count, out_count, 3, 1, 1),
            nn.BatchNorm2d(out_count),
            nn.GELU(),
            nn.Conv2d(out_count, out_count, 3, 1, 1),
            nn.BatchNorm2d(out_count),
            nn.GELU(),
            nn.MaxPool2d(2)
        )


class TransUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dc1 = DownConv(1, 16)
        self.dc2 = DownConv(16, 32)
        self.dc3 = DownConv(64, 128) # outputs 128 14x14 images...

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

    def forward(self, x):
        # 3 conv layers first...
        x1 = self.dc1(x)
        x2 = self.dc2(x1)
        x3 = self.dc3(x2) # 128 14x14 features... 
        # will now go into patch embedding...
        # and then transformer
        # and back out!
        
        
