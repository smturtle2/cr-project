# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from .attention import TransformerLayer


class MaskModule(nn.Module):
    def __init__(self, sar_channels, cloudy_channels, feature_channels, num_heads=4, patch_size=2, num_layers=2):
        super(MaskModule, self).__init__()
        self.embed = nn.Conv2d(sar_channels + cloudy_channels, feature_channels, kernel_size=1)
        self.patch_embed = nn.Conv2d(
            feature_channels,
            feature_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.layers = nn.ModuleList([
            TransformerLayer(feature_channels, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.expand = nn.Conv2d(feature_channels, feature_channels * patch_size * patch_size, kernel_size=1)
        self.upsample = nn.PixelShuffle(patch_size)
        self.out = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def _to_sequence(self, x):
        b, c, h, w = x.shape
        x = torch.einsum("b c h w -> b h w c", x)
        return x.reshape(b, h * w, c), h, w

    def _to_feature_map(self, x, h, w):
        b, _, c = x.shape
        x = x.reshape(b, h, w, c)
        return torch.einsum("b h w c -> b c h w", x)

    def forward(self, sar, cloudy):
        out = torch.cat((sar, cloudy), dim=1)
        out = self.embed(out)
        out = self.patch_embed(out)
        out, h, w = self._to_sequence(out)
        for layer in self.layers:
            out = layer(out)
        out = self._to_feature_map(out, h, w)
        out = self.expand(out)
        out = self.upsample(out)
        return self.out(out)
