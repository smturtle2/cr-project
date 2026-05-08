from __future__ import annotations

import torch
import torch.nn as nn

from ..module.attention import TransformerLayer, _sdpa


class Upsample(nn.Module):
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()
        self.expand = nn.Conv2d(dim, dim, kernel_size=1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.expand(feature))


class CommonGate(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Conv2d(dim, dim, kernel_size=1)

    @staticmethod
    def _to_tokens(feature: torch.Tensor) -> torch.Tensor:
        return feature.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def _to_feature(
        tokens: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        return tokens.transpose(1, 2).reshape(
            tokens.shape[0],
            tokens.shape[2],
            height,
            width,
        ).contiguous()

    def _reshape_heads(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, num_tokens, _ = tokens.shape
        return tokens.view(
            batch,
            num_tokens,
            self.heads,
            self.head_dim,
        ).transpose(1, 2).contiguous()

    def _restore_heads(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, _, num_tokens, _ = tokens.shape
        return tokens.transpose(1, 2).reshape(batch, num_tokens, self.dim).contiguous()

    def forward(
        self,
        target_feat: torch.Tensor,
        other_feat: torch.Tensor,
    ) -> torch.Tensor:
        height, width = target_feat.shape[-2:]
        query = self._reshape_heads(self.q_proj(self._to_tokens(target_feat)))
        key = self._reshape_heads(self.k_proj(self._to_tokens(other_feat)))
        value = self._reshape_heads(self.v_proj(self._to_tokens(other_feat)))
        context_tokens = _sdpa(query, key, value)
        context = self._to_feature(self._restore_heads(context_tokens), height, width)
        return self.gate_proj(context)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dim: int,
        num_layers: int,
        heads: int,
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=3, stride=2, padding=1)
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

        self.sar_encoder = Encoder(sar_channels, dim, num_layers, num_heads)
        self.cld_encoder = Encoder(cloudy_channels, dim, num_layers, num_heads)

        self.sar_common_gate = CommonGate(dim, heads=num_heads)
        self.cld_common_gate = CommonGate(dim, heads=num_heads)
        self.com_fuse = nn.Conv2d(self.up_dim * 2, self.up_dim * 2, kernel_size=1)

        self.sar_feat_up = Upsample(dim)
        self.cld_feat_up = Upsample(dim)
        self.sar_gate_up = Upsample(dim)
        self.cld_gate_up = Upsample(dim)

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor):
        sar_feat_l = self.sar_encoder(sar)
        cld_feat_l = self.cld_encoder(cloudy)

        sar_gate_l = self.sar_common_gate(sar_feat_l, cld_feat_l)
        cld_gate_l = self.cld_common_gate(cld_feat_l, sar_feat_l)

        sar_feat = self.sar_feat_up(sar_feat_l)
        cld_feat = self.cld_feat_up(cld_feat_l)
        sar_gate = torch.sigmoid(self.sar_gate_up(sar_gate_l))
        cld_gate = torch.sigmoid(self.cld_gate_up(cld_gate_l))

        sar_com = sar_gate * sar_feat
        cld_com = cld_gate * cld_feat
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
