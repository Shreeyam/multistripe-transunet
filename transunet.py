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
    def __init__(self, in_channels=1, patch_size=14, n_patches=128):
        self.patch_size = patch_size
        self.emb_size = (patch_size ** 2) * in_channels

        super().__init__()

        # no need to rearrange here... flatten will do the work for us
        self.projection = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear((patch_size ** 2) * in_channels, self.emb_size)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.emb_size))
        self.positions = nn.Parameter(torch.randn(n_patches, self.emb_size))

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


class Upscale(nn.Upsample):
    def __init__(self):
        super().__init__(scale_factor=2, mode='nearest')


class UpConv(nn.Sequential):
    def __init__(self, in_count, out_count):
        super().__init__(
            nn.Conv2d(in_count, out_count, 3, 1, 1),
            nn.BatchNorm2d(out_count),
            nn.GELU(),
            nn.Conv2d(out_count, out_count, 3, 1, 1),
            nn.BatchNorm2d(out_count),
            nn.GELU(),
        )


class TransUNet(nn.Module):
    def __init__(self):
        super().__init__()

        patch_size = 14
        filter_count = 128

        self.dc1 = DownConv(1, 16)
        self.dc2 = DownConv(16, 32)
        self.dc3 = DownConv(32, 64)
        self.dc4 = DownConv(64, filter_count)  # outputs 128 7x7 images...

        self.us4 = Upscale()
        self.us3 = Upscale()
        self.us2 = Upscale()
        self.us1 = Upscale()

        self.uc3 = UpConv(128, 32)
        self.uc2 = UpConv(64, 16)
        self.uc1 = UpConv(32, 16)

        self.embedding = Embedding()
        self.trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=patch_size ** 2, nhead=7)

        self.transformer = nn.TransformerEncoder(
            self.trans_encoder_layer, num_layers=6)

        self.transformer_convert = nn.Sequential(
            nn.Unflatten(dim=-1, unflattened_size=(patch_size, patch_size)),
            nn.Conv2d(filter_count, filter_count, 3, 1, 1),
            nn.BatchNorm2d(filter_count),
            nn.GELU(),
            nn.Conv2d(filter_count, filter_count//2, 3, 1, 1),
            nn.BatchNorm2d(filter_count//2),
            nn.GELU(),
        )

        self.outconv = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 3 conv layers first...
        x1 = self.dc1(x)
        x2 = self.dc2(x1)
        x3 = self.dc3(x2)
        x4 = self.dc4(x3)  # 128 14x14 features...
        # will now go into patch embedding...
        t_embed = self.embedding(x4)
        # and then transformer
        t_transformer = self.transformer(t_embed)
        # and back out! reshape to required size...
        # current shape: [b, 128, 196]

        y4 = self.transformer_convert(t_transformer)

        y3 = self.uc3(torch.cat((self.us4(y4), x3), 1))
        y2 = self.uc2(torch.cat((self.us3(y3), x2), 1))
        y1 = self.uc1(torch.cat((self.us2(y2), x1), 1))
        q = self.outconv(self.us1(y1))

        return q
