# -*- coding: utf-8 -*-

import torch.nn as nn

from .attention import TransformerLayer


class XSAEncoder(nn.Module):
    def __init__(self, sar_channels, feature_channels, num_heads=8, self_num_layers=2, patch_size=2):
        super(XSAEncoder, self).__init__()
        self.embed = nn.Conv2d(sar_channels, feature_channels, kernel_size=1)
        self.patch_embed = nn.Conv2d(
            feature_channels,
            feature_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.layers = nn.ModuleList([
            TransformerLayer(feature_channels, num_heads=num_heads, use_xsa=True)
            for _ in range(self_num_layers)
        ])

    def _to_sequence(self, x):
        b, c, h, w = x.shape
        return x.flatten(2).transpose(1, 2).contiguous(), h, w

    def forward(self, sar):
        out = self.embed(sar)
        out = self.patch_embed(out)
        out, h, w = self._to_sequence(out)
        for layer in self.layers:
            out = layer(out)
        return out, h, w


class CrossAttentionModule(nn.Module):
    def __init__(
        self,
        sar_channels,
        feature_channels,
        num_heads=8,
        patch_size=2,
        self_num_layers=2,
        cross_num_layers=2,
    ):
        super(CrossAttentionModule, self).__init__()
        self.patch_size = patch_size
        self.feature_channels = feature_channels
        self.sar_encoder = XSAEncoder(
            sar_channels,
            feature_channels,
            num_heads=num_heads,
            self_num_layers=self_num_layers,
            patch_size=patch_size,
        )
        self.feature_patch_embed = nn.Conv2d(
            feature_channels,
            feature_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cross_layers = nn.ModuleList([
            TransformerLayer(feature_channels, num_heads=num_heads, use_xsa=False)
            for _ in range(cross_num_layers)
        ])
        self.expand = nn.Conv2d(feature_channels, feature_channels * patch_size * patch_size, kernel_size=1)
        self.upsample = nn.PixelShuffle(patch_size)
        self.out_proj = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)

    def _to_sequence(self, x):
        b, c, h, w = x.shape
        return x.flatten(2).transpose(1, 2).contiguous(), h, w

    def _to_feature_map(self, x, h, w):
        b, _, c = x.shape
        return x.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, sar, feature):
        encoded_sar, h, w = self.sar_encoder(sar)
        out = self.feature_patch_embed(feature)
        out, _, _ = self._to_sequence(out)
        for layer in self.cross_layers:
            out = layer(out, encoded_sar)
        out = self._to_feature_map(out, h, w)
        out = self.expand(out)
        out = self.upsample(out)
        return self.out_proj(out)
