# -*- coding: utf-8 -*-

import torch.nn as nn

from ..gate import build_gate_estimator
from .cross_attention_module import CrossAttentionModule
from .mask_module import MaskModule


class BaseModule(nn.Module):
    def __init__(
        self,
        sar_channels,
        cloudy_channels,
        feature_channels,
        num_heads=4,
        patch_size=2,
        self_num_layers=2,
        cross_num_layers=2,
        gate_mode="mask",
        gate_feat_dim=32,
        gate_prior_weight=0.5,
        gate_scale=1.0,
    ):
        super(BaseModule, self).__init__()
        self.gate_mode = gate_mode
        self.gate_scale = float(gate_scale)
        self.gate_estimator = build_gate_estimator(
            gate_mode,
            sar_channels=sar_channels,
            optical_channels=cloudy_channels,
            feat_dim=gate_feat_dim,
            prior_weight=gate_prior_weight,
        )
        self.mask = MaskModule(
            sar_channels,
            cloudy_channels,
            feature_channels,
            num_heads=num_heads,
            patch_size=patch_size,
            num_layers=self_num_layers,
        )
        self.cross_attn = CrossAttentionModule(
            sar_channels,
            feature_channels,
            num_heads,
            patch_size=patch_size,
            self_num_layers=self_num_layers,
            cross_num_layers=cross_num_layers,
        )
        self.last_gate = None

    def forward(self, sar, cloudy, feature):
        if self.gate_estimator is None:
            gate = self.mask(sar, cloudy)
        else:
            gate = self.gate_estimator(sar, cloudy)
        self.last_gate = gate
        out = self.cross_attn(sar, feature)
        return feature + out * gate * self.gate_scale
