from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module.attention import MultiHeadAttention, TransformerLayer, _AmpRMSNorm


class _Residual3x3Block(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            ),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResizeConvUp(nn.Module):
    def __init__(self, in_channels: int, blocks: int = 2):
        super().__init__()
        if in_channels % 4 != 0:
            raise ValueError("in_channels must be divisible by 4")
        if blocks < 0:
            raise ValueError("blocks must be non-negative")

        self.in_channels = in_channels
        self.out_channels = in_channels // 4
        self.blocks = blocks
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            *[_Residual3x3Block(self.out_channels) for _ in range(blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        x = self.stem(x)
        return self.refine(x)


class _CrossAttentionGateExpert(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.target_norm = _AmpRMSNorm(dim)
        self.other_norm = _AmpRMSNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, num_heads=heads)

    @staticmethod
    def _to_tokens(feature: torch.Tensor) -> torch.Tensor:
        return feature.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def _to_feature(
        tokens: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        return (
            tokens.transpose(1, 2)
            .reshape(
                tokens.shape[0],
                tokens.shape[2],
                height,
                width,
            )
            .contiguous()
        )

    def forward(
        self,
        target_feat: torch.Tensor,
        other_feat: torch.Tensor,
    ) -> torch.Tensor:
        height, width = target_feat.shape[-2:]
        target_tokens = self.target_norm(self._to_tokens(target_feat))
        other_tokens = self.other_norm(self._to_tokens(other_feat))
        gate_tokens = self.cross_attn(target_tokens, other_tokens)
        return self._to_feature(gate_tokens, height, width)


class CommonGate(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.num_experts = 4
        self.experts = nn.ModuleList(
            [
                _CrossAttentionGateExpert(dim, heads=heads)
                for _ in range(self.num_experts)
            ]
        )
        self.router = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, self.num_experts, kernel_size=1),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def forward(
        self,
        target_feat: torch.Tensor,
        other_feat: torch.Tensor,
    ) -> torch.Tensor:
        router_logits = self.router(target_feat)
        router_weights = torch.softmax(router_logits.float(), dim=1).to(
            dtype=router_logits.dtype
        )

        expert_logits = torch.stack(
            [expert(target_feat, other_feat) for expert in self.experts],
            dim=1,
        )
        return (expert_logits * router_weights.unsqueeze(2)).sum(dim=1)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dim: int,
        num_layers: int,
        heads: int,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                dim,
                kernel_size=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GELU(),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode="replicate",
            ),
        )
        self.blocks = nn.ModuleList(
            [TransformerLayer(dim, num_heads=heads) for _ in range(num_layers)]
        )

    @staticmethod
    def _to_tokens(feature: torch.Tensor) -> torch.Tensor:
        return feature.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def _to_feature(
        tokens: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        return (
            tokens.transpose(1, 2)
            .reshape(tokens.shape[0], tokens.shape[2], height, width)
            .contiguous()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.proj(x)
        height, width = feature.shape[-2:]
        tokens = self._to_tokens(feature)
        for block in self.blocks:
            tokens = block(tokens)
        return self._to_feature(tokens, height, width)


# Feature Decomposition Transformer
class FDT(nn.Module):
    def __init__(
        self,
        sar_channels=2,
        cloudy_channels=13,
        dim=256,
        num_layers=2,
        num_heads=4,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if dim % 4 != 0:
            raise ValueError("dim must be divisible by 4")
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero")

        self.sar_channels = sar_channels
        self.cloudy_channels = cloudy_channels
        self.dim = dim
        self.num_layers = num_layers
        self.heads = num_heads
        self.num_heads = num_heads
        self.up_dim = dim // 4
        self.common_dim = dim // 2

        self.sar_encoder = Encoder(sar_channels, dim, num_layers, num_heads)
        self.cld_encoder = Encoder(cloudy_channels, dim, num_layers, num_heads)

        self.sar_common_gate = CommonGate(dim, heads=num_heads)
        self.cld_common_gate = CommonGate(dim, heads=num_heads)
        self.com_fuse = nn.Conv2d(self.up_dim * 2, self.common_dim, kernel_size=1)
        self.up = ResizeConvUp(dim)

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor):
        sar_feat_l = self.sar_encoder(sar)
        cld_feat_l = self.cld_encoder(cloudy)

        sar_gate_l = torch.sigmoid(self.sar_common_gate(sar_feat_l, cld_feat_l))
        cld_gate_l = torch.sigmoid(self.cld_common_gate(cld_feat_l, sar_feat_l))

        sar_com_l = sar_gate_l * sar_feat_l
        cld_com_l = cld_gate_l * cld_feat_l

        sar_feat = self.up(sar_feat_l)
        cld_feat = self.up(cld_feat_l)
        sar_com = self.up(sar_com_l)
        cld_com = self.up(cld_com_l)
        sar_comp = sar_feat - sar_com
        cld_comp = cld_feat - cld_com

        com_fused = self.com_fuse(torch.cat((sar_com, cld_com), dim=1))
        output = torch.cat((com_fused, sar_comp, cld_comp), dim=1)
        return (
            output,
            sar_com,
            cld_com,
            sar_comp,
            cld_comp,
        )
