# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class SARLightEncoder(nn.Module):
    def __init__(
        self,
        sar_channels: int = 2,
        feature_channels: int = 256,
        num_blocks: int = 2,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(sar_channels, feature_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        ]
        for _ in range(num_blocks - 1):
            layers += [
                nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=True),
                nn.GELU(),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, sar: torch.Tensor) -> torch.Tensor:
        return self.net(sar)


class CrossModalMDTA(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, bias: bool = False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias)
        self.kv_proj = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, f_opt: torch.Tensor, f_sar: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = f_opt.shape
        head_dim = channels // self.num_heads

        q = self.q_dwconv(self.q_proj(f_opt))
        kv = self.kv_dwconv(self.kv_proj(f_sar))
        k, v = kv.chunk(2, dim=1)

        q = q.view(batch, self.num_heads, head_dim, height * width)
        k = k.view(batch, self.num_heads, head_dim, height * width)
        v = v.view(batch, self.num_heads, head_dim, height * width)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.contiguous().view(batch, channels, height, width)
        return self.project_out(out)


class GDFN(nn.Module):
    def __init__(self, dim: int, expansion: float = 2.0, bias: bool = False):
        super().__init__()
        hidden = int(dim * expansion)
        self.proj_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            kernel_size=3,
            padding=1,
            groups=hidden * 2,
            bias=bias,
        )
        self.proj_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dwconv(self.proj_in(x))
        x1, x2 = x.chunk(2, dim=1)
        return self.proj_out(F.gelu(x1) * x2)


class CrossModalBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ffn_expansion: float = 2.0,
        bias: bool = False,
    ):
        super().__init__()
        self.norm_opt = LayerNorm2d(dim)
        self.norm_sar = LayerNorm2d(dim)
        self.attn = CrossModalMDTA(dim, num_heads=num_heads, bias=bias)
        self.norm_ffn = LayerNorm2d(dim)
        self.ffn = GDFN(dim, expansion=ffn_expansion, bias=bias)

    def forward(self, f_opt: torch.Tensor, f_sar: torch.Tensor) -> torch.Tensor:
        f_opt = f_opt + self.attn(self.norm_opt(f_opt), self.norm_sar(f_sar))
        f_opt = f_opt + self.ffn(self.norm_ffn(f_opt))
        return f_opt


class CrossAttentionModule(nn.Module):
    def __init__(
        self,
        sar_channels,
        feature_channels,
        num_heads=4,
        patch_size=2,
        self_num_layers=2,
        cross_num_layers=2,
    ):
        super().__init__()
        del patch_size, cross_num_layers
        self.sar_encoder = SARLightEncoder(
            sar_channels=sar_channels,
            feature_channels=feature_channels,
            num_blocks=self_num_layers,
        )
        self.cross_modal = CrossModalBlock(
            dim=feature_channels,
            num_heads=num_heads,
        )

    def forward(self, sar: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        f_sar = self.sar_encoder(sar)
        return self.cross_modal(feature, f_sar)
