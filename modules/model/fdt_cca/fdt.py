from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..fdt.fdt import Residual3x3Block, ResizeConvUp
from ..fdt.fdt import CommonEncoder as FeatureEncoder


class JointEncoder(FeatureEncoder):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        heads: int,
    ):
        super().__init__(dim, num_layers, heads)
        self.proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.GELU(),
            Residual3x3Block(dim),
            Residual3x3Block(dim),
        )

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        joint = torch.cat((first, second), dim=1).contiguous()
        return super().forward(self.proj(joint))


class FeatureBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        heads: int,
    ):
        super().__init__()
        self.sar_feat_encoder = FeatureEncoder(dim, num_layers, heads)
        self.cld_feat_encoder = FeatureEncoder(dim, num_layers, heads)

    def forward(
        self,
        sar: torch.Tensor,
        cld: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sar_feat_encoder(sar), self.cld_feat_encoder(cld)


class CommonBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        heads: int,
    ):
        super().__init__()
        self.joint_encoder = JointEncoder(dim, num_layers, heads)
        self.sar_feat_encoder = FeatureEncoder(dim, num_layers, heads)
        self.cld_feat_encoder = FeatureEncoder(dim, num_layers, heads)

    def forward(
        self,
        sar: torch.Tensor,
        cld: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joint = self.joint_encoder(sar, cld)
        return self.sar_feat_encoder(joint), self.cld_feat_encoder(joint)


def _validate_extractor_dims(
    dims: tuple[int, ...] | list[int],
    heads: int,
) -> tuple[int, ...]:
    dims = tuple(dims)
    if len(dims) < 2:
        raise ValueError("extractor_dims must contain at least two levels")
    if heads <= 0:
        raise ValueError("heads must be greater than zero")
    for dim in dims:
        if dim <= 0:
            raise ValueError("extractor dims must be positive")
        if dim % heads != 0:
            raise ValueError("extractor dims must be divisible by heads")
    return dims


def _conv_block(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            padding_mode="replicate",
        ),
        nn.GELU(),
        nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
        ),
        nn.GELU(),
    )


class ResizeProjectUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GELU(),
        )
        self.refine = Residual3x3Block(out_channels)

    def forward(
        self,
        x: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        x = self.proj(x)
        return self.refine(x)


class Extractor(nn.Module):
    def __init__(
        self,
        sar_channels: int,
        cloudy_channels: int,
        dims: tuple[int, ...] | list[int] = (128, 256, 512),
        num_layers: int = 2,
        heads: int = 4,
        block_cls: type[nn.Module] = FeatureBlock,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero")

        dims = _validate_extractor_dims(dims, heads)
        self.dims = dims
        self.num_levels = len(dims)
        self.block_cls = block_cls

        self.sar_stem = _conv_block(sar_channels, dims[0])
        self.cld_stem = _conv_block(cloudy_channels, dims[0])
        self.sar_downs = nn.ModuleList(
            [
                _conv_block(dims[i], dims[i + 1], stride=2)
                for i in range(len(dims) - 1)
            ]
        )
        self.cld_downs = nn.ModuleList(
            [
                _conv_block(dims[i], dims[i + 1], stride=2)
                for i in range(len(dims) - 1)
            ]
        )
        self.encoder_blocks = nn.ModuleList(
            [block_cls(dim, num_layers, heads) for dim in dims[1:]]
        )
        self.sar_ups = nn.ModuleList(
            [
                ResizeProjectUp(dims[i], dims[i - 1])
                for i in range(len(dims) - 1, 0, -1)
            ]
        )
        self.cld_ups = nn.ModuleList(
            [
                ResizeProjectUp(dims[i], dims[i - 1])
                for i in range(len(dims) - 1, 0, -1)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                block_cls(dims[i], num_layers, heads)
                for i in range(len(dims) - 2, 0, -1)
            ]
        )
        self.sar_out = _conv_block(dims[0], dims[0])
        self.cld_out = _conv_block(dims[0], dims[0])

    def forward(
        self,
        sar: torch.Tensor,
        cld: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sar = self.sar_stem(sar)
        cld = self.cld_stem(cld)
        sar_levels = [sar]
        cld_levels = [cld]

        last_down_index = len(self.sar_downs) - 1
        encoder_steps = zip(
            self.sar_downs,
            self.cld_downs,
            self.encoder_blocks,
        )
        for index, (sar_down, cld_down, block) in enumerate(encoder_steps):
            sar = sar_down(sar)
            cld = cld_down(cld)
            sar, cld = block(sar, cld)
            if index != last_down_index:
                sar_levels.append(sar)
                cld_levels.append(cld)

        for level, sar_up, cld_up, block in zip(
            range(self.num_levels - 2, 0, -1),
            self.sar_ups[:-1],
            self.cld_ups[:-1],
            self.decoder_blocks,
        ):
            sar = (
                sar_up(sar, size=sar_levels[level].shape[-2:])
                + sar_levels[level]
            )
            cld = (
                cld_up(cld, size=cld_levels[level].shape[-2:])
                + cld_levels[level]
            )
            sar, cld = block(sar, cld)

        sar = (
            self.sar_ups[-1](sar, size=sar_levels[0].shape[-2:])
            + sar_levels[0]
        )
        cld = (
            self.cld_ups[-1](cld, size=cld_levels[0].shape[-2:])
            + cld_levels[0]
        )
        return self.sar_out(sar), self.cld_out(cld)


class ResizeConvUpHalf(ResizeConvUp):
    def __init__(self, in_channels: int, blocks: int = 2):
        super().__init__(in_channels, blocks=blocks)
        self.out_channels = in_channels // 2
        self.refine = nn.Conv2d(
            in_channels,
            self.out_channels,
            kernel_size=1,
        )


class FDT_CCA(nn.Module):
    def __init__(
        self,
        sar_channels=2,
        cloudy_channels=13,
        dim=256,
        num_layers=2,
        num_heads=4,
        extractor_dims: tuple[int, ...] | list[int] = (128, 256, 512),
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero")

        extractor_dims = _validate_extractor_dims(extractor_dims, num_heads)
        if extractor_dims[0] * 2 != dim:
            raise ValueError("extractor_dims[0] * 2 must match dim")

        self.sar_channels = sar_channels
        self.cloudy_channels = cloudy_channels
        self.dim = dim
        self.num_layers = num_layers
        self.heads = num_heads
        self.num_heads = num_heads
        self.extractor_dims = extractor_dims
        self.up_dim = extractor_dims[0]
        self.common_dim = extractor_dims[0]
        self.feature_extractor = Extractor(
            sar_channels,
            cloudy_channels,
            dims=extractor_dims,
            num_layers=num_layers,
            heads=num_heads,
            block_cls=FeatureBlock,
        )
        self.common_extractor = Extractor(
            self.up_dim,
            self.up_dim,
            dims=extractor_dims,
            num_layers=num_layers,
            heads=num_heads,
            block_cls=CommonBlock,
        )
        self.com_fuse = nn.Conv2d(self.up_dim * 2, self.common_dim, kernel_size=1)

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor):
        sar_feat, cld_feat = self.feature_extractor(sar, cloudy)
        sar_com, cld_com = self.common_extractor(sar_feat, cld_feat)
        sar_comp = sar_feat - sar_com
        cld_comp = cld_feat - cld_com

        com_fused = self.com_fuse(torch.cat((sar_com, cld_com), dim=1))
        output = torch.cat((com_fused, sar_comp), dim=1)
        return (
            output,
            sar_com,
            cld_com,
            sar_comp,
            cld_comp,
        )
