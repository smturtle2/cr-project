# -*- coding: utf-8 -*-

import torch.nn as nn

from .cross_attention_module import CrossAttentionModule
from .mask_module import MaskModule


class BaseModule(nn.Module):
    def __init__(
        self,
        sar_channels,
        cloudy_channels,
        feature_channels,
        num_heads=8,
        patch_size=2,
        self_num_layers=2,
        cross_num_layers=2,
    ):
        super(BaseModule, self).__init__()
        self.mask = MaskModule(sar_channels, cloudy_channels, feature_channels)
        self.cross_attn = CrossAttentionModule(
            sar_channels,
            feature_channels,
            num_heads,
            patch_size=patch_size,
            self_num_layers=self_num_layers,
            cross_num_layers=cross_num_layers,
        )

    def forward(self, sar, cloudy, feature):
        mask = self.mask(sar, cloudy, feature)
        out = self.cross_attn(sar, feature)
        return feature + out * mask
