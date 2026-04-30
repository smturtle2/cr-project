# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel


class CrossAttentionModule(nn.Module):
    def __init__(self, sar_channels, feature_channels, num_heads=8):
        super(CrossAttentionModule, self).__init__()
        self.num_heads = num_heads
        self.head_dim = feature_channels // num_heads
        self.q_proj = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.k_proj = nn.Conv2d(sar_channels, feature_channels, kernel_size=1)
        self.v_proj = nn.Conv2d(sar_channels, feature_channels, kernel_size=1)
        self.out_proj = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)

    def forward(self, sar, feature):
        if sar.shape[-2:] != feature.shape[-2:]:
            sar = F.interpolate(sar, size=feature.shape[-2:], mode="bilinear", align_corners=True)

        q = self.q_proj(feature)
        k = self.k_proj(sar)
        v = self.v_proj(sar)

        b, c, h, w = q.shape
        num_tokens = h * w
        q = q.flatten(2).transpose(1, 2).contiguous()
        k = k.flatten(2).transpose(1, 2).contiguous()
        v = v.flatten(2).transpose(1, 2).contiguous()

        q = q.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(
                q.to(torch.float16),
                k.to(torch.float16),
                v.to(torch.float16),
                dropout_p=0.0,
            )
        out = out.to(feature.dtype)
        out = out.transpose(1, 2).reshape(b, num_tokens, c)
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return self.out_proj(out)
