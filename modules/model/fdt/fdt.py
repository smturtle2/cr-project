from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module.attention import TransformerLayer


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


class FeatureTransformerBase(nn.Module):
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


class Encoder(FeatureTransformerBase):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.proj(x)
        height, width = feature.shape[-2:]
        tokens = self._to_tokens(feature)
        for block in self.blocks:
            tokens = block(tokens)
        return self._to_feature(tokens, height, width)


class CommonEncoder(FeatureTransformerBase):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        heads: int,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerLayer(dim, num_heads=heads) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        tokens = self._to_tokens(x)
        for block in self.blocks:
            tokens = block(tokens)
        return self._to_feature(tokens, height, width)


class BranchEncoder(CommonEncoder):
    pass


class JointEncoder(FeatureTransformerBase):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        heads: int,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            ),
        )
        self.blocks = nn.ModuleList(
            [TransformerLayer(dim, num_heads=heads) for _ in range(num_layers)]
        )

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        feature = self.proj(torch.cat((first, second), dim=1))
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

        self.joint_encoder = JointEncoder(dim, num_layers, num_heads)
        self.feat1_encoder = BranchEncoder(dim, num_layers, num_heads)
        self.feat2_encoder = BranchEncoder(dim, num_layers, num_heads)
        self.feat1_common_encoder = CommonEncoder(dim, num_layers, num_heads)
        self.feat2_common_encoder = CommonEncoder(dim, num_layers, num_heads)
        self.com_fuse = nn.Conv2d(self.up_dim * 2, self.common_dim, kernel_size=1)
        self.up = ResizeConvUp(dim)

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor):
        sar_base_l = self.sar_encoder(sar)
        cld_base_l = self.cld_encoder(cloudy)
        joint_l = self.joint_encoder(sar_base_l, cld_base_l)

        feat1_l = self.feat1_encoder(joint_l)
        feat2_l = self.feat2_encoder(joint_l)
        feat1_com_l = self.feat1_common_encoder(feat1_l)
        feat2_com_l = self.feat2_common_encoder(feat2_l)

        feat1 = self.up(feat1_l)
        feat2 = self.up(feat2_l)
        feat1_com = self.up(feat1_com_l)
        feat2_com = self.up(feat2_com_l)
        feat1_comp = feat1 - feat1_com
        feat2_comp = feat2 - feat2_com

        com_fused = self.com_fuse(torch.cat((feat1_com, feat2_com), dim=1))
        output = torch.cat((com_fused, feat1_comp, feat2_comp), dim=1)
        return (
            output,
            feat1_com,
            feat2_com,
            feat1_comp,
            feat2_comp,
        )
