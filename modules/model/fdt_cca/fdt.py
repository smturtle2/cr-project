from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..fdt.fdt import CommonEncoder as FeatureEncoder
from ..fdt.fdt import Residual3x3Block, ResizeConvUp


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


class SarStem(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dim: int,
        blocks: int = 2,
    ):
        super().__init__()
        if blocks < 0:
            raise ValueError("blocks must be non-negative")

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                dim,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[Residual3x3Block(dim) for _ in range(blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.proj(x))


class CloudyStem(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dim: int,
        blocks: int = 2,
    ):
        super().__init__()
        if blocks < 0:
            raise ValueError("blocks must be non-negative")

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                dim,
                kernel_size=1,
            ),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[Residual3x3Block(dim) for _ in range(blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.proj(x))


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        x = self.proj(x)
        return self.refine(x)


class DownLevels(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dims: tuple[int, ...],
    ):
        super().__init__()
        self.stem = _conv_block(in_channels, dims[0])
        self.downs = nn.ModuleList(
            [_conv_block(dims[i], dims[i + 1], stride=2) for i in range(len(dims) - 1)]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        levels = [x]

        for down in self.downs:
            x = down(x)
            levels.append(x)
        return levels


class UpLevels(nn.Module):
    def __init__(
        self,
        dims: tuple[int, ...],
        num_layers: int,
        heads: int,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [FeatureEncoder(dim, num_layers, heads) for dim in reversed(dims[1:])]
        )
        self.ups = nn.ModuleList(
            [ResizeProjectUp(dims[i], dims[i - 1]) for i in range(len(dims) - 1, 0, -1)]
        )
        self.recons = nn.ModuleList(
            [ResizeProjectUp(dims[i + 1], dims[i]) for i in range(len(dims) - 1)]
        )

    def forward(self, levels: list[torch.Tensor]) -> torch.Tensor:
        residuals = [
            current - recon(next_level)
            for current, next_level, recon in zip(
                levels[:-1],
                levels[1:],
                self.recons,
            )
        ]
        x = levels[-1]

        for level, block, up in zip(
            range(len(levels) - 1, 0, -1),
            self.blocks,
            self.ups,
        ):
            x = block(x)
            x = up(x) + residuals[level - 1]
        return x


class ExtractorLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        dims: tuple[int, ...] | list[int] = (128, 256, 512),
        num_layers: int = 2,
        heads: int = 4,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero")

        dims = _validate_extractor_dims(dims, heads)
        if dims[0] != dim:
            raise ValueError("dim must match extractor_dims[0]")

        self.dim = dim
        self.dims = dims
        self.num_levels = len(dims)

        self.down = DownLevels(dim, dims)
        self.up = UpLevels(dims, num_layers, heads)
        self.out = Residual3x3Block(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        levels = self.down(x)
        x = self.up(levels)
        return self.out(x)


class Extractor(nn.Module):
    def __init__(
        self,
        dim: int,
        dims: tuple[int, ...] | list[int] = (128, 256, 512),
        layer_count: int = 1,
        num_layers: int = 2,
        heads: int = 4,
    ):
        super().__init__()
        if layer_count <= 0:
            raise ValueError("layer_count must be greater than zero")

        dims = _validate_extractor_dims(dims, heads)
        if dims[0] != dim:
            raise ValueError("dim must match extractor_dims[0]")

        self.dim = dim
        self.dims = dims
        self.layer_count = layer_count
        self.layers = nn.ModuleList(
            [
                ExtractorLayer(
                    dim,
                    dims=dims,
                    num_layers=num_layers,
                    heads=heads,
                )
                for _ in range(layer_count)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


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
        num_layers: int = 2,
        feature_extractor_layers: int = 2,
        num_heads: int = 4,
        extractor_dims: tuple[int, ...] | list[int] = (128, 256, 512),
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero")
        if feature_extractor_layers <= 0:
            raise ValueError("feature_extractor_layers must be greater than zero")

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
        self.feature_extractor_layers = feature_extractor_layers
        self.up_dim = extractor_dims[0]
        self.sar_stem = SarStem(sar_channels, self.up_dim)
        self.cld_stem = CloudyStem(cloudy_channels, self.up_dim)
        self.sar_extractor = Extractor(
            self.up_dim,
            dims=extractor_dims,
            layer_count=feature_extractor_layers,
            num_layers=num_layers,
            heads=num_heads,
        )
        self.cld_extractor = Extractor(
            self.up_dim,
            dims=extractor_dims,
            layer_count=feature_extractor_layers,
            num_layers=num_layers,
            heads=num_heads,
        )
        self.cld_common_extractor = Extractor(
            self.up_dim,
            dims=extractor_dims,
            num_layers=num_layers,
            heads=num_heads,
        )

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor):
        # feature extraction
        sar_feat = self.sar_extractor(self.sar_stem(sar))
        cld_feat = self.cld_extractor(self.cld_stem(cloudy))

        # Cloudy-only decomposition. SAR stays as the shared scene reference.
        cld_com = self.cld_common_extractor(cld_feat)
        cld_comp = cld_feat - cld_com

        output = torch.cat((sar_feat, cld_com), dim=1)

        return (
            output,
            sar_feat,
            cld_com,
            cld_comp,
        )
