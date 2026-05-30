from __future__ import annotations

import torch
import torch.nn as nn

from ..module.attention import TransformerLayer


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
            ChannelAttention(out_channels),
            SpatialAttention(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PixelShuffleUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                4 * out_channels,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
                bias=False,
            ),
            nn.PixelShuffle(2),
            ChannelAttention(out_channels),
            SpatialAttention(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Residual3x3Block(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
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
        return x + self.net(x)


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
        self.out_channels = out_channels if out_channels is not None else in_channels // 4
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
        return (
            output,
            sar_com,
            cld_com,
            sar_comp,
            cld_comp,
        )
