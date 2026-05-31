from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module.attention import TransformerLayer


class RMSNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.square().mean(dim=1, keepdim=True)
        return x * torch.rsqrt(rms + self.eps) * self.weight


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        hidden = max(channels // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="reflect",
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat((avg_out, max_out), dim=1)))
        return attn * x


class PixelUnshuffleDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        if out_channels % 4 != 0:
            raise ValueError("out_channels must be divisible by 4")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels // 4,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
                bias=False,
            ),
            nn.PixelUnshuffle(2),
            Residual3x3Block(out_channels),
            ChannelAttention(out_channels),
            SpatialAttention(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# DySample+
class DySample(nn.Module):
    def __init__(self, channels: int, scale: int = 2, groups: int = 4):
        super().__init__()
        if scale <= 1:
            raise ValueError("scale must be greater than one")
        if channels < groups or channels % groups != 0:
            raise ValueError("channels must be divisible by groups")

        self.channels = channels
        self.scale = scale
        self.groups = groups
        self.offset = nn.Conv2d(channels, 2 * groups * scale**2, kernel_size=1)
        self.scope = nn.Conv2d(
            channels,
            2 * groups * scale**2,
            kernel_size=1,
            bias=False,
        )
        nn.init.normal_(self.offset.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.scope.weight)
        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self) -> torch.Tensor:
        h = (
            torch.arange(
                (-self.scale + 1) / 2,
                (self.scale - 1) / 2 + 1,
            )
            / self.scale
        )
        return (
            torch.stack(torch.meshgrid(h, h, indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        offset = offset.view(batch, 2, -1, height, width)

        coords_h = torch.arange(height, dtype=x.dtype, device=x.device) + 0.5
        coords_w = torch.arange(width, dtype=x.dtype, device=x.device) + 0.5
        coords = (
            torch.stack(
                torch.meshgrid(coords_w, coords_h, indexing="xy"),
            )
            .unsqueeze(1)
            .unsqueeze(0)
        )
        normalizer = x.new_tensor([width, height]).view(1, 2, 1, 1, 1)
        coords = 2.0 * (coords + offset) / normalizer - 1.0
        coords = F.pixel_shuffle(
            coords.view(batch, -1, height, width),
            self.scale,
        )
        coords = (
            coords.view(
                batch,
                2,
                -1,
                self.scale * height,
                self.scale * width,
            )
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        return F.grid_sample(
            x.reshape(batch * self.groups, -1, height, width),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(batch, -1, self.scale * height, self.scale * width)


class PixelShuffleUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.proj = nn.Sequential(
            DySample(in_channels),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
                bias=False,
            ),
            Residual3x3Block(out_channels),
            ChannelAttention(out_channels),
            SpatialAttention(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Residual3x3Block(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = RMSNorm2d(channels)
        self.net = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
            ),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
            ),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


_Residual3x3Block = Residual3x3Block


class ResizeConvUp(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
    ):
        super().__init__()
        if out_channels is None and in_channels % 4 != 0:
            raise ValueError("in_channels must be divisible by 4")

        self.in_channels = in_channels
        self.out_channels = (
            out_channels if out_channels is not None else in_channels // 4
        )
        self.up = PixelShuffleUp(in_channels, self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


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
                padding_mode="reflect",
            ),
            nn.GELU(),
            PixelUnshuffleDown(dim, dim),
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

        self.sar_common_encoder = CommonEncoder(dim, num_layers, num_heads)
        self.cld_common_encoder = CommonEncoder(dim, num_layers, num_heads)
        self.com_fuse = nn.Conv2d(self.up_dim * 2, self.common_dim, kernel_size=1)
        self.up = ResizeConvUp(dim)

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor):
        sar_feat_l = self.sar_encoder(sar)
        cld_feat_l = self.cld_encoder(cloudy)

        sar_com_l = self.sar_common_encoder(sar_feat_l)
        cld_com_l = self.cld_common_encoder(cld_feat_l)

        sar_feat = self.up(sar_feat_l)
        cld_feat = self.up(cld_feat_l)
        sar_com = self.up(sar_com_l)
        cld_com = self.up(cld_com_l)
        sar_comp = sar_feat - sar_com
        cld_comp = cld_feat - cld_com

        com_fused = self.com_fuse(torch.cat((sar_com, cld_com), dim=1))
        output = torch.cat((com_fused, sar_comp, cld_comp), dim=1)
        return {
            "feature": output,
            "sar_common": sar_com,
            "cloudy_common": cld_com,
            "sar_component": sar_comp,
            "cloudy_component": cld_comp,
        }
