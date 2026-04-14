from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _reshape_spatial_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    batch, channels, height, width = x.shape
    head_dim = channels // heads
    return x.view(batch, heads, head_dim, height, width).permute(0, 1, 3, 4, 2)


def _flatten_spatial_heads(x: torch.Tensor) -> torch.Tensor:
    batch, heads, height, width, head_dim = x.shape
    return x.reshape(batch, heads, height * width, head_dim)


def _restore_spatial_heads(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    batch, heads, _, head_dim = x.shape
    return x.reshape(batch, heads, height, width, head_dim).permute(0, 1, 4, 2, 3).reshape(
        batch, heads * head_dim, height, width
    )


def _exclude_self_value_component(
    attn_out: torch.Tensor,
    self_value: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    self_value_unit = F.normalize(self_value, dim=-1, eps=eps)
    return attn_out - (attn_out * self_value_unit).sum(dim=-1, keepdim=True) * self_value_unit


def _validate_dropout_prob(name: str, value: float) -> float:
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0), got {value}")
    return value


class _AttentionCore(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        self_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query.ndim != 4:
            raise ValueError("attention core expects query to have shape (batch, heads, tokens, head_dim)")
        if key.ndim == query.ndim:
            scores = (query * self.scale) @ key.transpose(-2, -1)
        elif key.ndim == query.ndim + 1:
            scores = torch.einsum("bhtd,bhtkd->bhtk", query, key) * self.scale
        else:
            raise ValueError("attention core expects dense or gathered key/value tensors")

        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        attn = scores.softmax(dim=-1)

        if value.ndim == query.ndim:
            out = attn @ value
        elif value.ndim == query.ndim + 1:
            out = torch.einsum("bhtk,bhtkd->bhtd", attn, value)
        else:
            raise ValueError("attention core expects dense or gathered key/value tensors")

        if self_value is not None:
            out = _exclude_self_value_component(out, self_value)
        return out


class RMSNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.square().mean(dim=1, keepdim=True)
        return x * torch.rsqrt(rms + self.eps) * self.weight


class DWConvFFN(nn.Module):
    def __init__(self, dim: int, expansion: int = 2):
        super().__init__()
        hidden_dim = dim * expansion
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BilinearUp2x(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if size is not None:
            return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)


class NeighborhoodAttn(nn.Module):
    def __init__(self, dim: int, heads: int, neighborhood_size: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        if neighborhood_size <= 0 or neighborhood_size % 2 == 0:
            raise ValueError("neighborhood_size must be a positive odd integer")
        self.dim = dim
        self.heads = heads
        self.neighborhood_size = neighborhood_size
        self.radius = neighborhood_size // 2
        self.scale = (dim // heads) ** -0.5
        self.core = _AttentionCore(scale=self.scale)
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        offsets = torch.arange(-self.radius, self.radius + 1, dtype=torch.long)
        offset_grid = torch.stack(torch.meshgrid(offsets, offsets, indexing="ij"), dim=-1)
        self.register_buffer("offsets", offset_grid.view(-1, 2), persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        is_self_attn = key is None and value is None
        if is_self_attn:
            key = query
            value = query
        elif key is None or value is None:
            raise ValueError("neighborhood attn expects both key and value when used as cross attn")
        if query.shape[-2:] != key.shape[-2:] or key.shape[-2:] != value.shape[-2:]:
            raise ValueError("neighborhood attn expects matching spatial sizes")

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        batch, _, height, width = q.shape
        head_dim = self.dim // self.heads
        q_heads = _reshape_spatial_heads(q, self.heads)
        k_heads = _reshape_spatial_heads(k, self.heads)
        v_heads = _reshape_spatial_heads(v, self.heads)

        y_coords = torch.arange(height, device=q.device, dtype=torch.long)
        x_coords = torch.arange(width, device=q.device, dtype=torch.long)
        base_y, base_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

        neighbor_y = base_y[..., None] + self.offsets[:, 0]
        neighbor_x = base_x[..., None] + self.offsets[:, 1]
        valid = (
            (neighbor_y >= 0)
            & (neighbor_y < height)
            & (neighbor_x >= 0)
            & (neighbor_x < width)
        )
        neighbor_y = neighbor_y.clamp(0, height - 1)
        neighbor_x = neighbor_x.clamp(0, width - 1)

        batch_idx = torch.arange(batch, device=q.device)[:, None, None, None, None]
        head_idx = torch.arange(self.heads, device=q.device)[None, :, None, None, None]
        neighbor_y = neighbor_y[None, None, ...]
        neighbor_x = neighbor_x[None, None, ...]
        valid = valid[None, None, ...]

        k_neighbors = k_heads[batch_idx, head_idx, neighbor_y, neighbor_x]
        v_neighbors = v_heads[batch_idx, head_idx, neighbor_y, neighbor_x]
        query_tokens = _flatten_spatial_heads(q_heads)
        key_tokens = k_neighbors.reshape(batch, self.heads, height * width, -1, head_dim)
        value_tokens = v_neighbors.reshape(batch, self.heads, height * width, -1, head_dim)
        valid_mask = valid.reshape(1, 1, height * width, -1)
        self_value = _flatten_spatial_heads(v_heads) if is_self_attn else None

        out = self.core(
            query_tokens,
            key_tokens,
            value_tokens,
            valid_mask=valid_mask,
            self_value=self_value,
        )
        out = _restore_spatial_heads(out, height, width)
        return self.out_proj(out)


class GlobalSelfAttn(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.core = _AttentionCore(scale=self.scale)
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_heads = _flatten_spatial_heads(_reshape_spatial_heads(self.q_proj(x), self.heads))
        k_heads = _flatten_spatial_heads(_reshape_spatial_heads(self.k_proj(x), self.heads))
        v_heads = _flatten_spatial_heads(_reshape_spatial_heads(self.v_proj(x), self.heads))

        out = self.core(q_heads, k_heads, v_heads, self_value=v_heads)
        out = _restore_spatial_heads(out, *x.shape[-2:])
        return self.out_proj(out)


class LocalBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        neighborhood_size: int,
        ffn_expansion: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        dropout = _validate_dropout_prob("dropout", dropout)
        self.z_attn_norm = RMSNorm2d(dim)
        self.h_attn_norm = RMSNorm2d(dim)
        self.attn = NeighborhoodAttn(
            dim=dim,
            heads=heads,
            neighborhood_size=neighborhood_size,
        )
        self.attn_dropout = nn.Dropout(p=dropout)
        self.ffn_norm = RMSNorm2d(dim)
        self.ffn = DWConvFFN(dim=dim, expansion=ffn_expansion)
        self.ffn_dropout = nn.Dropout(p=dropout)

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        h_norm = self.h_attn_norm(h)
        z = z + self.attn_dropout(self.attn(self.z_attn_norm(z), h_norm, h_norm))
        z = z + self.ffn_dropout(self.ffn(self.ffn_norm(z)))
        return z


class GlobalBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        ffn_expansion: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        dropout = _validate_dropout_prob("dropout", dropout)
        self.z_attn_norm = RMSNorm2d(dim)
        self.attn = GlobalSelfAttn(dim=dim, heads=heads)
        self.attn_dropout = nn.Dropout(p=dropout)
        self.z_downsample = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
        self.upsample = BilinearUp2x()
        self.ffn_norm = RMSNorm2d(dim)
        self.ffn = DWConvFFN(dim=dim, expansion=ffn_expansion)
        self.ffn_dropout = nn.Dropout(p=dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if min(z.shape[-2:]) < 2:
            z_coarse = z
        else:
            z_coarse = self.z_downsample(z)

        z_coarse_norm = self.z_attn_norm(z_coarse)
        attn = self.attn(z_coarse_norm)
        if attn.shape[-2:] != z.shape[-2:]:
            attn = self.upsample(attn, size=z.shape[-2:])
        z = z + self.attn_dropout(attn)
        z = z + self.ffn_dropout(self.ffn(self.ffn_norm(z)))
        return z


class LCRWrapperBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        neighborhood_size: int,
        ffn_expansion: int,
        local_block_count: int,
        global_block_count: int,
        block_dropout: float = 0.0,
    ):
        super().__init__()
        if local_block_count <= 0:
            raise ValueError("local_block_count must be greater than zero")
        if global_block_count <= 0:
            raise ValueError("global_block_count must be greater than zero")
        block_dropout = _validate_dropout_prob("block_dropout", block_dropout)

        self.local_blocks = nn.ModuleList(
            [
                LocalBlock(
                    dim=dim,
                    heads=heads,
                    neighborhood_size=neighborhood_size,
                    ffn_expansion=ffn_expansion,
                    dropout=block_dropout,
                )
                for _ in range(local_block_count)
            ]
        )
        self.global_blocks = nn.ModuleList(
            [
                GlobalBlock(
                    dim=dim,
                    heads=heads,
                    ffn_expansion=ffn_expansion,
                    dropout=block_dropout,
                )
                for _ in range(global_block_count)
            ]
        )

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        for local_block in self.local_blocks:
            z = local_block(z, h)
        for global_block in self.global_blocks:
            z = global_block(z)
        return z


class ResDWBlock(nn.Module):
    def __init__(self, dim: int, ffn_expansion: int, dropout: float = 0.0):
        super().__init__()
        dropout = _validate_dropout_prob("dropout", dropout)
        self.conv_norm = RMSNorm2d(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv_dropout = nn.Dropout(p=dropout)
        self.ffn_norm = RMSNorm2d(dim)
        self.ffn = DWConvFFN(dim=dim, expansion=ffn_expansion)
        self.ffn_dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.conv_dropout(self.conv(self.act(self.conv_norm(x))))
        x = x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))
        return x


class PointwiseResBlock(nn.Module):
    def __init__(self, dim: int, expansion: int, dropout: float = 0.0):
        super().__init__()
        dropout = _validate_dropout_prob("dropout", dropout)
        hidden_dim = dim * expansion
        self.norm = RMSNorm2d(dim)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
        )
        self.residual_dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual_dropout(self.net(self.norm(x)))


class AttnStem(nn.Module):
    def __init__(self, in_channels: int, dim: int, heads: int, neighborhood_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=1)
        self.downsample = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
        self.attn_norm = RMSNorm2d(dim)
        self.attn = NeighborhoodAttn(
            dim=dim,
            heads=heads,
            neighborhood_size=neighborhood_size,
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.proj(x))
        x = self.act(self.downsample(x))
        return x + self.attn(self.attn_norm(x))


class LatentDecoder(nn.Module):
    def __init__(self, dim: int, ffn_expansion: int, dropout: float = 0.0):
        super().__init__()
        dropout = _validate_dropout_prob("dropout", dropout)
        self.full_dim = max(dim // 2, 1)
        self.block_latent = ResDWBlock(dim=dim, ffn_expansion=ffn_expansion, dropout=dropout)
        self.project = nn.Conv2d(dim, self.full_dim, kernel_size=1)
        self.upsample = BilinearUp2x()
        self.block_full = PointwiseResBlock(
            dim=self.full_dim,
            expansion=ffn_expansion,
            dropout=dropout,
        )
        self.out = nn.Conv2d(self.full_dim, self.full_dim, kernel_size=1)

    def forward(self, latent: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        decoded = self.block_latent(latent)
        decoded = self.project(decoded)
        decoded = self.upsample(decoded, size=output_size)
        decoded = self.block_full(decoded)
        return self.out(decoded)


class LCR(nn.Module):
    def __init__(
        self,
        *,
        sar_channels: int = 2,
        opt_channels: int = 13,
        dim: int = 64,
        num_blocks: int = 6,
        heads: int = 4,
        neighborhood_size: int = 7,
        ffn_expansion: int = 2,
        local_block_count: int = 1,
        global_block_count: int = 1,
        mask_bias_init: float = -2.0,
        block_dropout: float = 0.08,
        decoder_dropout: float = 0.05,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        if neighborhood_size <= 0 or neighborhood_size % 2 == 0:
            raise ValueError("neighborhood_size must be a positive odd integer")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be greater than zero")
        if local_block_count <= 0:
            raise ValueError("local_block_count must be greater than zero")
        if global_block_count <= 0:
            raise ValueError("global_block_count must be greater than zero")
        block_dropout = _validate_dropout_prob("block_dropout", block_dropout)
        decoder_dropout = _validate_dropout_prob("decoder_dropout", decoder_dropout)

        self.sar_channels = sar_channels
        self.opt_channels = opt_channels
        self.dim = dim
        self.num_blocks = num_blocks
        self.local_block_count = local_block_count
        self.global_block_count = global_block_count
        self.neighborhood_size = neighborhood_size
        self.block_dropout = block_dropout
        self.decoder_dropout = decoder_dropout

        self.sar_stem = AttnStem(
            in_channels=sar_channels,
            dim=dim,
            heads=heads,
            neighborhood_size=neighborhood_size,
        )
        self.hsi_stem = AttnStem(
            in_channels=opt_channels,
            dim=dim,
            heads=heads,
            neighborhood_size=neighborhood_size,
        )

        self.wrapper_blocks = nn.ModuleList(
            [
                LCRWrapperBlock(
                    dim=dim,
                    heads=heads,
                    neighborhood_size=neighborhood_size,
                    ffn_expansion=ffn_expansion,
                    local_block_count=local_block_count,
                    global_block_count=global_block_count,
                    block_dropout=block_dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        self.candidate_decoder = LatentDecoder(
            dim=dim,
            ffn_expansion=ffn_expansion,
            dropout=decoder_dropout,
        )
        self.mask_decoder = LatentDecoder(
            dim=dim,
            ffn_expansion=ffn_expansion,
            dropout=decoder_dropout,
        )

        self.candidate_head = nn.Sequential(
            nn.Conv2d(self.candidate_decoder.full_dim, self.candidate_decoder.full_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.candidate_decoder.full_dim, opt_channels, kernel_size=1),
        )

        self.mask_out = nn.Conv2d(self.mask_decoder.full_dim, 1, kernel_size=1)

        nn.init.constant_(self.mask_out.bias, mask_bias_init)

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor) -> torch.Tensor:
        if sar.shape[0] != cloudy.shape[0] or sar.shape[-2:] != cloudy.shape[-2:]:
            raise ValueError("sar and cloudy inputs must share the same batch and spatial size")
        if sar.shape[1] != self.sar_channels:
            raise ValueError(f"expected sar to have {self.sar_channels} channels, got {sar.shape[1]}")
        if cloudy.shape[1] != self.opt_channels:
            raise ValueError(
                f"expected cloudy to have {self.opt_channels} channels, got {cloudy.shape[1]}"
            )
        if sar.shape[-2] % 4 != 0 or sar.shape[-1] % 4 != 0:
            raise ValueError("LCR expects spatial dimensions divisible by 4")

        z0 = self.sar_stem(sar)
        h = self.hsi_stem(cloudy)
        z = z0

        for wrapper_block in self.wrapper_blocks:
            z = wrapper_block(z, h)

        output_size = cloudy.shape[-2:]
        candidate = self.candidate_head(self.candidate_decoder(z, output_size=output_size))
        mask_logits = self.mask_out(self.mask_decoder(z, output_size=output_size))
        mask = torch.sigmoid(mask_logits)

        return (1.0 - mask) * cloudy + mask * candidate
