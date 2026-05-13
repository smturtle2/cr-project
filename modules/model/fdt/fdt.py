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
        self.residuals = nn.Sequential(
            *[_Residual3x3Block(in_channels) for _ in range(blocks)]
        )
        self.refine = nn.Conv2d(
            in_channels,
            self.out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        x = self.residuals(x)
        return self.refine(x)


class PairedCompBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: int = 4):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.sar_query_norm = _AmpRMSNorm(dim)
        self.sar_key_norm = _AmpRMSNorm(dim)
        self.sar_value_norm = _AmpRMSNorm(dim)
        self.cld_query_norm = _AmpRMSNorm(dim)
        self.cld_key_norm = _AmpRMSNorm(dim)
        self.cld_value_norm = _AmpRMSNorm(dim)
        self.sar_cross_attn = MultiHeadAttention(dim, num_heads=heads)
        self.cld_cross_attn = MultiHeadAttention(dim, num_heads=heads)

        hidden_dim = dim * mlp_ratio
        self.sar_mlp_norm = _AmpRMSNorm(dim)
        self.cld_mlp_norm = _AmpRMSNorm(dim)
        self.sar_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, dim),
        )
        self.cld_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(
        self,
        sar_state: torch.Tensor,
        cld_state: torch.Tensor,
        sar_anchor: torch.Tensor,
        cld_anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sar_cross = sar_state + self.sar_cross_attn(
            self.sar_query_norm(sar_state),
            self.sar_key_norm(cld_state),
            self.sar_value_norm(sar_anchor),
        )
        cld_cross = cld_state + self.cld_cross_attn(
            self.cld_query_norm(cld_state),
            self.cld_key_norm(sar_state),
            self.cld_value_norm(cld_anchor),
        )

        sar_out = sar_cross + self.sar_mlp(self.sar_mlp_norm(sar_cross))
        cld_out = cld_cross + self.cld_mlp(self.cld_mlp_norm(cld_cross))
        return sar_out, cld_out


class PairedCompTransformer(nn.Module):
    def __init__(self, dim: int, num_layers: int, heads: int):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.heads = heads
        self.blocks = nn.ModuleList(
            [PairedCompBlock(dim, heads=heads) for _ in range(num_layers)]
        )
        self.sar_comp_head = nn.Linear(dim, dim)
        self.cld_comp_head = nn.Linear(dim, dim)

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

    def forward(
        self,
        sar_feat: torch.Tensor,
        cld_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = sar_feat.shape[-2:]
        sar_anchor = self._to_tokens(sar_feat)
        cld_anchor = self._to_tokens(cld_feat)
        sar_tokens = sar_anchor
        cld_tokens = cld_anchor
        for block in self.blocks:
            sar_tokens, cld_tokens = block(
                sar_tokens,
                cld_tokens,
                sar_anchor,
                cld_anchor,
            )
        sar_comp = self.sar_comp_head(sar_tokens)
        cld_comp = self.cld_comp_head(cld_tokens)
        return (
            self._to_feature(sar_comp, height, width),
            self._to_feature(cld_comp, height, width),
        )


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

        self.comp_extractor = PairedCompTransformer(
            dim,
            num_layers=num_layers,
            heads=num_heads,
        )
        self.com_fuse = nn.Conv2d(self.up_dim * 2, self.common_dim, kernel_size=1)
        self.up = ResizeConvUp(dim)

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor):
        sar_feat_l = self.sar_encoder(sar)
        cld_feat_l = self.cld_encoder(cloudy)

        sar_comp_l, cld_comp_l = self.comp_extractor(sar_feat_l, cld_feat_l)

        sar_feat = self.up(sar_feat_l)
        cld_feat = self.up(cld_feat_l)
        sar_comp = self.up(sar_comp_l)
        cld_comp = self.up(cld_comp_l)
        sar_com = sar_feat - sar_comp
        cld_com = cld_feat - cld_comp

        com_fused = self.com_fuse(torch.cat((sar_com, cld_com), dim=1))
        output = torch.cat((com_fused, sar_comp, cld_comp), dim=1)
        return (
            output,
            sar_com,
            cld_com,
            sar_comp,
            cld_comp,
        )
