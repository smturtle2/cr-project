from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..fdt.fdt import CommonEncoder as FeatureEncoder
from ..fdt.fdt import Residual3x3Block, ResizeConvUp


class RMSNorm2d(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().square().mean(dim=1, keepdim=True)
        scale = torch.rsqrt(rms + self.eps).to(dtype=x.dtype)
        return x * scale


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


class Extractor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dims: tuple[int, ...] | list[int] = (128, 256, 512),
        num_layers: int = 2,
        heads: int = 4,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero")

        dims = _validate_extractor_dims(dims, heads)
        self.dims = dims
        self.num_levels = len(dims)

        self.down = DownLevels(in_channels, dims)
        self.up = UpLevels(dims, num_layers, heads)
        self.out = Residual3x3Block(dims[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        levels = self.down(x)
        x = self.up(levels)
        return self.out(x)


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
        self.sar_extractor = Extractor(
            sar_channels,
            dims=extractor_dims,
            num_layers=num_layers,
            heads=num_heads,
        )
        self.cld_extractor = Extractor(
            cloudy_channels,
            dims=extractor_dims,
            num_layers=num_layers,
            heads=num_heads,
        )
        self.joint_extractor = Extractor(
            self.up_dim * 2,
            dims=extractor_dims,
            num_layers=num_layers,
            heads=num_heads,
        )
        self.sar_joint_norm = RMSNorm2d()
        self.cld_joint_norm = RMSNorm2d()
        self.sar_common_extractor = Extractor(
            self.up_dim,
            dims=extractor_dims,
            num_layers=num_layers,
            heads=num_heads,
        )
        self.cld_common_extractor = Extractor(
            self.up_dim,
            dims=extractor_dims,
            num_layers=num_layers,
            heads=num_heads,
        )
        self.com_fuse = nn.Conv2d(self.up_dim * 2, self.common_dim, kernel_size=1)

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor):
        # feature extraction
        sar_feat = self.sar_extractor(sar)
        cld_feat = self.cld_extractor(cloudy)

        # joint feature extraction
        joint_input = torch.cat(
            (self.sar_joint_norm(sar_feat), self.cld_joint_norm(cld_feat)),
            dim=1,
        )
        joint = self.joint_extractor(joint_input)
        sar_feat = sar_feat + joint
        cld_feat = cld_feat + joint

        # common and complementary extraction
        sar_com = self.sar_common_extractor(sar_feat)
        cld_com = self.cld_common_extractor(cld_feat)
        sar_comp = sar_feat - sar_com
        cld_comp = cld_feat - cld_com

        # fusion and output
        com_fused = self.com_fuse(torch.cat((sar_com, cld_com), dim=1))
        output = torch.cat((com_fused, sar_comp), dim=1)

        return (
            output,
            sar_com,
            cld_com,
            sar_comp,
            cld_comp,
        )
