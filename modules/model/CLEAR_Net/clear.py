from __future__ import annotations

import math
import numbers

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

from .ca_flash import _sdpa


class AmpRMSNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps=None,
        elementwise_affine=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if weight is not None and weight.dtype != x.dtype:
            weight = weight.to(dtype=x.dtype)
        return F.rms_norm(x, self.normalized_shape, weight, self.eps)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def _restore_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, tokens, _ = x.shape
        return x.transpose(1, 2).reshape(batch, tokens, self.dim)

    def forward(
        self,
        query: torch.Tensor,
        key_source: torch.Tensor | None = None,
        value_source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        is_self_attention = key_source is None and value_source is None
        if key_source is None:
            key_source = query
        if value_source is None:
            value_source = key_source

        q = self._reshape_heads(self.q_proj(query))
        k = self._reshape_heads(self.k_proj(key_source))
        v = self._reshape_heads(self.v_proj(value_source))
        out = _sdpa(q, k, v)

        if is_self_attention:
            out = out.float()
            v_norm = F.normalize(v.float(), dim=-1, eps=1e-6)
            out = out - (out * v_norm).sum(dim=-1, keepdim=True) * v_norm

        out = self._restore_heads(out.to(query.dtype))
        return self.out_proj(out)


class TransformerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = AmpRMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads)
        self.norm2 = AmpRMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.ReLU(True),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_source)
        return x + self.mlp(self.norm2(x))


class RMSNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.square().mean(dim=1, keepdim=True)
        return x * torch.rsqrt(rms + self.eps) * self.weight


class Residual3x3Block(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = RMSNorm2d(channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="reflect"),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class RefineHead(nn.Module):
    def __init__(self, channels: int, out_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            Residual3x3Block(channels),
            Residual3x3Block(channels),
        )
        self.norm = RMSNorm2d(channels)
        self.proj = nn.Conv2d(channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(self.body(x)))


class SpectralMaskRouter(nn.Module):
    def __init__(self, channels: int, out_channels: int, num_routes: int = 32):
        super().__init__()
        self.num_routes = num_routes
        self.out_channels = out_channels
        self.route_head = RefineHead(channels, num_routes)
        self.channel_routes = nn.Parameter(torch.empty(num_routes, out_channels))
        init.normal_(self.channel_routes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        route_logits = self.route_head(x)
        route_weights = torch.softmax(route_logits, dim=1)
        channel_routes = self.channel_routes.to(dtype=route_weights.dtype)
        mask_logits = torch.einsum("bkhw,kc->bchw", route_weights, channel_routes)
        return {
            "mask": torch.sigmoid(mask_logits),
            "route_weights": route_weights,
        }


class SampleDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DySample(nn.Module):
    def __init__(self, channels: int, scale: int = 2, groups: int = 4):
        super().__init__()
        self.scale = scale
        self.groups = groups
        self.offset = nn.Conv2d(channels, 2 * groups * scale**2, kernel_size=1)
        self.scope = nn.Conv2d(channels, 2 * groups * scale**2, kernel_size=1, bias=False)
        nn.init.normal_(self.offset.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.scope.weight)
        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self) -> torch.Tensor:
        h = torch.arange(
            (-self.scale + 1) / 2,
            (self.scale - 1) / 2 + 1,
        ) / self.scale
        return (
            torch.stack(torch.meshgrid(h, h, indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        offset = offset.reshape(batch, 2, -1, height, width)

        coords_h = torch.arange(height, dtype=x.dtype, device=x.device) + 0.5
        coords_w = torch.arange(width, dtype=x.dtype, device=x.device) + 0.5
        coords = torch.stack(torch.meshgrid(coords_w, coords_h, indexing="xy"))
        coords = coords.unsqueeze(1).unsqueeze(0)
        normalizer = x.new_tensor([width, height]).view(1, 2, 1, 1, 1)
        coords = 2.0 * (coords + offset) / normalizer - 1.0
        coords = F.pixel_shuffle(coords.reshape(batch, -1, height, width), self.scale)
        coords = (
            coords.reshape(batch, 2, -1, self.scale * height, self.scale * width)
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
        ).reshape(batch, -1, self.scale * height, self.scale * width)


class SampleUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FeatureTransformerBase(nn.Module):
    @staticmethod
    def _to_tokens(feature: torch.Tensor) -> torch.Tensor:
        return feature.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def _to_feature(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height, width).contiguous()


class FeatureEncoder(FeatureTransformerBase):
    def __init__(self, dim: int, num_layers: int, heads: int):
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


def _conv_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            padding_mode="reflect",
        ),
        nn.GELU(),
        nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        ),
        nn.GELU(),
    )


class Stem(nn.Module):
    def __init__(self, in_channels: int, dim: int, blocks: int = 3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                dim,
                kernel_size=7,
                padding=3,
                padding_mode="reflect",
            ),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[Residual3x3Block(dim) for _ in range(blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.proj(x))


class DownLevels(nn.Module):
    def __init__(self, in_channels: int, dims: tuple[int, ...]):
        super().__init__()
        self.stem = _conv_block(in_channels, dims[0])
        self.downs = nn.ModuleList(
            [SampleDown(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        levels = [x]
        for down in self.downs:
            x = down(x)
            levels.append(x)
        return levels


class UpLevels(nn.Module):
    def __init__(self, dims: tuple[int, ...], num_layers: int, heads: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [FeatureEncoder(dim, num_layers, heads) for dim in reversed(dims[1:])]
        )
        self.ups = nn.ModuleList(
            [SampleUp(dims[i], dims[i - 1]) for i in range(len(dims) - 1, 0, -1)]
        )
        self.up_norms = nn.ModuleList(
            [RMSNorm2d(dims[i]) for i in range(len(dims) - 1, 0, -1)]
        )
        self.recons = nn.ModuleList(
            [SampleUp(dims[i + 1], dims[i]) for i in range(len(dims) - 1)]
        )

    def forward(self, levels: list[torch.Tensor]) -> torch.Tensor:
        residuals = [
            current - recon(next_level)
            for current, next_level, recon in zip(levels[:-1], levels[1:], self.recons)
        ]
        x = levels[-1]

        for level, block, up_norm, up in zip(
            range(len(levels) - 1, 0, -1),
            self.blocks,
            self.up_norms,
            self.ups,
            strict=False,
        ):
            x = block(x)
            x = up(up_norm(x)) + residuals[level - 1]
        return x


class ExtractorLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        dims: tuple[int, ...] | list[int] = (64, 128, 256),
        num_layers: int = 2,
        heads: int = 4,
    ):
        super().__init__()
        dims = tuple(dims)
        self.down = DownLevels(dim, dims)
        self.up = UpLevels(dims, num_layers, heads)
        self.out = Residual3x3Block(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.up(self.down(x)))


class Extractor(nn.Module):
    def __init__(
        self,
        dim: int,
        dims: tuple[int, ...] | list[int] = (64, 128, 256),
        layer_count: int = 1,
        num_layers: int = 2,
        heads: int = 4,
    ):
        super().__init__()
        dims = tuple(dims)
        self.layers = nn.ModuleList(
            [
                ExtractorLayer(dim, dims=dims, num_layers=num_layers, heads=heads)
                for _ in range(layer_count)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
