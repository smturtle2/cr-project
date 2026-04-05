from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _reshape_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    batch, tokens, channels = x.shape
    head_dim = channels // heads
    return x.view(batch, tokens, heads, head_dim).transpose(1, 2)


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


class ShuffleUp2x(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.scale_factor = 2
        self.expand = nn.Conv2d(channels, channels * (self.scale_factor**2), kernel_size=1)
        self.shuffle = nn.PixelShuffle(self.scale_factor)
        self.act = nn.GELU()
        self.fuse = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        target_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        upsampled = self.shuffle(self.expand(x))
        if target_size is not None:
            target_h, target_w = target_size
            out_h, out_w = upsampled.shape[-2:]
            if target_h > out_h or target_w > out_w:
                raise ValueError("target_size must not exceed the native 2x upsampled size")
            upsampled = upsampled[..., :target_h, :target_w]
        return self.fuse(self.act(upsampled))


class NeighborhoodCrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int, neighborhood_size: int):
        super().__init__()
        if neighborhood_size <= 0 or neighborhood_size % 2 == 0:
            raise ValueError("neighborhood_size must be a positive odd integer")
        self.dim = dim
        self.heads = heads
        self.neighborhood_size = neighborhood_size
        self.radius = neighborhood_size // 2
        self.scale = (dim // heads) ** -0.5
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        offsets = torch.arange(-self.radius, self.radius + 1, dtype=torch.long)
        offset_grid = torch.stack(torch.meshgrid(offsets, offsets, indexing="ij"), dim=-1)
        self.register_buffer("offsets", offset_grid.view(-1, 2), persistent=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if query.shape[-2:] != key.shape[-2:] or key.shape[-2:] != value.shape[-2:]:
            raise ValueError("neighborhood cross attention expects matching spatial sizes")

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        batch, _, height, width = q.shape
        head_dim = self.dim // self.heads
        q_heads = q.view(batch, self.heads, head_dim, height, width).permute(0, 1, 3, 4, 2)
        k_heads = k.view(batch, self.heads, head_dim, height, width).permute(0, 1, 3, 4, 2)
        v_heads = v.view(batch, self.heads, head_dim, height, width).permute(0, 1, 3, 4, 2)

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

        attn = (q_heads.unsqueeze(-2) * k_neighbors).sum(dim=-1) * self.scale
        attn = attn.masked_fill(~valid, torch.finfo(attn.dtype).min)
        attn = attn.softmax(dim=-1)
        out = (attn.unsqueeze(-1) * v_neighbors).sum(dim=-2)
        out = out.permute(0, 1, 4, 2, 3).reshape(batch, self.dim, height, width)
        return self.out_proj(out)


class GlobalCrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if query.shape[-2:] != key.shape[-2:] or key.shape[-2:] != value.shape[-2:]:
            raise ValueError("global cross attention expects matching spatial sizes")

        q = self.q_proj(query).flatten(2).transpose(1, 2)
        k = self.k_proj(key).flatten(2).transpose(1, 2)
        v = self.v_proj(value).flatten(2).transpose(1, 2)

        q_heads = _reshape_heads(q, self.heads)
        k_heads = _reshape_heads(k, self.heads)
        v_heads = _reshape_heads(v, self.heads)

        attn = (q_heads * self.scale) @ k_heads.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        out = attn @ v_heads
        out = out.transpose(1, 2).reshape(query.shape[0], -1, self.dim)
        out = out.transpose(1, 2).reshape_as(query)
        return self.out_proj(out)


class LocalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, neighborhood_size: int, ffn_expansion: int):
        super().__init__()
        self.z_attn_norm = RMSNorm2d(dim)
        self.h_attn_norm = RMSNorm2d(dim)
        self.attn = NeighborhoodCrossAttention(
            dim=dim,
            heads=heads,
            neighborhood_size=neighborhood_size,
        )
        self.ffn_norm = RMSNorm2d(dim)
        self.ffn = DWConvFFN(dim=dim, expansion=ffn_expansion)

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        h_norm = self.h_attn_norm(h)
        z = z + self.attn(self.z_attn_norm(z), h_norm, h_norm)
        z = z + self.ffn(self.ffn_norm(z))
        return z


class GlobalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_expansion: int):
        super().__init__()
        self.z_attn_norm = RMSNorm2d(dim)
        self.h_attn_norm = RMSNorm2d(dim)
        self.attn = GlobalCrossAttention(dim=dim, heads=heads)
        self.ffn_norm = RMSNorm2d(dim)
        self.ffn = DWConvFFN(dim=dim, expansion=ffn_expansion)

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if min(z.shape[-2:]) < 2 or min(h.shape[-2:]) < 2:
            z_coarse = z
            h_coarse = h
        else:
            z_coarse = F.max_pool2d(z, kernel_size=2, stride=2)
            h_coarse = F.max_pool2d(h, kernel_size=2, stride=2)

        h_coarse_norm = self.h_attn_norm(h_coarse)
        attn = self.attn(
            self.z_attn_norm(z_coarse),
            h_coarse_norm,
            h_coarse_norm,
        )
        attn = F.interpolate(attn, size=z.shape[-2:], mode="bilinear", align_corners=False)
        z = z + attn
        z = z + self.ffn(self.ffn_norm(z))
        return z


class ResDWBlock(nn.Module):
    def __init__(self, dim: int, ffn_expansion: int):
        super().__init__()
        self.conv_norm = RMSNorm2d(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.ffn_norm = RMSNorm2d(dim)
        self.ffn = DWConvFFN(dim=dim, expansion=ffn_expansion)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.conv(self.act(self.conv_norm(x)))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class PointwiseResBlock(nn.Module):
    def __init__(self, dim: int, expansion: int):
        super().__init__()
        hidden_dim = dim * expansion
        self.norm = RMSNorm2d(dim)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class ConvStem(nn.Module):
    def __init__(self, in_channels: int, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=1)
        self.down1 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
        self.down2 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.act(self.proj(x))
        skip_h2 = self.act(self.down1(x))
        latent = self.down2(skip_h2)
        return latent, skip_h2


class ReconstructionTrunk(nn.Module):
    def __init__(self, dim: int, ffn_expansion: int):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.upsample_h2 = ShuffleUp2x(dim)
        self.merge = nn.Conv2d(dim * 3, dim, kernel_size=1)
        self.block_h2 = ResDWBlock(dim=dim, ffn_expansion=ffn_expansion)
        self.upsample_full = ShuffleUp2x(dim)
        self.proj_full = nn.Conv2d(dim, dim, kernel_size=1)
        self.block_full = PointwiseResBlock(dim=dim, expansion=ffn_expansion)
        self.out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(
        self,
        z_f: torch.Tensor,
        *,
        sar_skip_h2: torch.Tensor,
        hsi_skip_h2: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        u0 = self.proj(z_f)
        u1 = self.upsample_h2(u0, target_size=sar_skip_h2.shape[-2:])
        u1 = self.merge(torch.cat([u1, sar_skip_h2, hsi_skip_h2], dim=1))
        u1 = self.block_h2(u1)

        u2 = self.upsample_full(u1, target_size=output_size)
        u2 = self.proj_full(u2)
        u2 = self.block_full(u2)
        return self.out(u2)


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
        mask_bias_init: float = -2.0,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        if neighborhood_size <= 0 or neighborhood_size % 2 == 0:
            raise ValueError("neighborhood_size must be a positive odd integer")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be greater than zero")

        self.sar_channels = sar_channels
        self.opt_channels = opt_channels
        self.dim = dim
        self.num_blocks = num_blocks
        self.neighborhood_size = neighborhood_size

        self.sar_stem = ConvStem(in_channels=sar_channels, dim=dim)
        self.hsi_stem = ConvStem(in_channels=opt_channels, dim=dim)

        self.local_blocks = nn.ModuleList(
            [
                LocalBlock(
                    dim=dim,
                    heads=heads,
                    neighborhood_size=neighborhood_size,
                    ffn_expansion=ffn_expansion,
                )
                for _ in range(num_blocks)
            ]
        )
        self.global_blocks = nn.ModuleList(
            [
                GlobalBlock(
                    dim=dim,
                    heads=heads,
                    ffn_expansion=ffn_expansion,
                )
                for _ in range(num_blocks)
            ]
        )

        self.reconstruction = ReconstructionTrunk(dim=dim, ffn_expansion=ffn_expansion)

        self.candidate_head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, opt_channels, kernel_size=1),
        )

        mask_hidden = max(dim // 2, 1)
        self.mask_head_coarse = nn.Sequential(
            nn.Conv2d(dim, mask_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mask_hidden, 1, kernel_size=1),
        )
        self.mask_head_refine = nn.Sequential(
            nn.Conv2d(dim + 2, mask_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mask_hidden, 1, kernel_size=1),
        )

        nn.init.constant_(self.mask_head_coarse[-1].bias, mask_bias_init)
        nn.init.constant_(self.mask_head_refine[-1].bias, mask_bias_init)

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor) -> torch.Tensor:
        if sar.shape[0] != cloudy.shape[0] or sar.shape[-2:] != cloudy.shape[-2:]:
            raise ValueError("sar and cloudy inputs must share the same batch and spatial size")
        if sar.shape[1] != self.sar_channels:
            raise ValueError(f"expected sar to have {self.sar_channels} channels, got {sar.shape[1]}")
        if cloudy.shape[1] != self.opt_channels:
            raise ValueError(
                f"expected cloudy to have {self.opt_channels} channels, got {cloudy.shape[1]}"
            )

        z, sar_skip_h2 = self.sar_stem(sar)
        h, hsi_skip_h2 = self.hsi_stem(cloudy)

        for local_block, global_block in zip(self.local_blocks, self.global_blocks, strict=True):
            z = local_block(z, h)
            z = global_block(z, h)

        reconstruction = self.reconstruction(
            z,
            sar_skip_h2=sar_skip_h2,
            hsi_skip_h2=hsi_skip_h2,
            output_size=cloudy.shape[-2:],
        )
        candidate = self.candidate_head(reconstruction)
        mask_coarse_logits = self.mask_head_coarse(z)
        mask_coarse = torch.sigmoid(mask_coarse_logits)
        mask_coarse_up = F.interpolate(
            mask_coarse,
            size=cloudy.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        diff = (candidate - cloudy).abs().mean(dim=1, keepdim=True)
        mask_logits = self.mask_head_refine(torch.cat([mask_coarse_up, reconstruction, diff], dim=1))
        mask = torch.sigmoid(mask_logits)

        return (1.0 - mask) * cloudy + mask * candidate
