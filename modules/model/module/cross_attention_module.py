# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel


class XSAEncoderLayer(nn.Module):
    def __init__(self, feature_channels, num_heads=8):
        super(XSAEncoderLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = feature_channels // num_heads
        self.q_proj = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.k_proj = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.v_proj = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.out_proj = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.norm1 = nn.GroupNorm(1, feature_channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels * 4, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(feature_channels * 4, feature_channels, kernel_size=1),
        )
        self.norm2 = nn.GroupNorm(1, feature_channels)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        b, c, h, w = q.shape
        num_tokens = h * w
        q = q.flatten(2).transpose(1, 2).contiguous()
        k = k.flatten(2).transpose(1, 2).contiguous()
        v = v.flatten(2).transpose(1, 2).contiguous()

        q = q.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        q_attn = q.to(torch.float16)
        k_attn = k.to(torch.float16)
        v_attn = v.to(torch.float16)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(q_attn, k_attn, v_attn, dropout_p=0.0)

        v_norm = F.normalize(v_attn, dim=-1)
        out = out - (out * v_norm).sum(dim=-1, keepdim=True) * v_norm
        out = out.to(x.dtype)
        out = out.transpose(1, 2).reshape(b, num_tokens, c)
        out = out.transpose(1, 2).reshape(b, c, h, w)

        x = self.norm1(x + self.out_proj(out))
        x = self.norm2(x + self.ffn(x))
        return x


class XSAEncoder(nn.Module):
    def __init__(self, sar_channels, feature_channels, num_heads=8, num_layers=2):
        super(XSAEncoder, self).__init__()
        self.embed = nn.Conv2d(sar_channels, feature_channels, kernel_size=1)
        self.layers = nn.ModuleList([
            XSAEncoderLayer(feature_channels, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, sar):
        out = self.embed(sar)
        for layer in self.layers:
            out = layer(out)
        return out


class CrossAttentionModule(nn.Module):
    def __init__(self, sar_channels, feature_channels, num_heads=8):
        super(CrossAttentionModule, self).__init__()
        self.num_heads = num_heads
        self.head_dim = feature_channels // num_heads
        self.sar_encoder = XSAEncoder(sar_channels, feature_channels, num_heads=num_heads, num_layers=2)
        self.q_proj = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.k_proj = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.v_proj = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.out_proj = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)

    def forward(self, sar, feature):
        if sar.shape[-2:] != feature.shape[-2:]:
            sar = F.interpolate(sar, size=feature.shape[-2:], mode="bilinear", align_corners=True)

        encoded_sar = self.sar_encoder(sar)
        q = self.q_proj(feature)
        k = self.k_proj(encoded_sar)
        v = self.v_proj(encoded_sar)

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
