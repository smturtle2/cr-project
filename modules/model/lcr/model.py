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


class _AttnCore(nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        self_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_dtype = query.dtype
        if query.is_cuda and query.dtype == torch.float32:
            query = query.to(torch.bfloat16).contiguous()
            key = key.to(torch.bfloat16).contiguous()
            value = value.to(torch.bfloat16).contiguous()
        elif query.is_cuda:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
        ).to(dtype=output_dtype)

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
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attn(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.core = _AttnCore()
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor | None = None,
        *,
        exclude_self_value: bool = False,
    ) -> torch.Tensor:
        if context is None:
            context = query

        q = self.q_proj(query)
        k = self.k_proj(context)
        v = self.v_proj(context)

        q_heads = _flatten_spatial_heads(_reshape_spatial_heads(q, self.heads))
        k_heads = _flatten_spatial_heads(_reshape_spatial_heads(k, self.heads))
        v_heads = _flatten_spatial_heads(_reshape_spatial_heads(v, self.heads))

        self_value = v_heads if exclude_self_value else None
        out = self.core(q_heads, k_heads, v_heads, self_value=self_value)
        out = _restore_spatial_heads(out, *query.shape[-2:])
        return self.out_proj(out)


class AttnBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        ffn_expansion: int,
        dropout: float = 0.0,
        use_context_norm: bool = False,
        exclude_self_value: bool = False,
    ):
        super().__init__()
        dropout = _validate_dropout_prob("dropout", dropout)
        self.attn_norm = RMSNorm2d(dim)
        self.context_norm = RMSNorm2d(dim) if use_context_norm else None
        self.exclude_self_value = exclude_self_value
        self.attn = Attn(dim=dim, heads=heads)
        self.attn_dropout = nn.Dropout(p=dropout)
        self.ffn_norm = RMSNorm2d(dim)
        self.ffn = DWConvFFN(dim=dim, expansion=ffn_expansion)
        self.ffn_dropout = nn.Dropout(p=dropout)

    def forward(self, z: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if context is not None and self.context_norm is not None:
            context = self.context_norm(context)
        z = z + self.attn_dropout(
            self.attn(
                self.attn_norm(z),
                context,
                exclude_self_value=self.exclude_self_value,
            )
        )
        z = z + self.ffn_dropout(self.ffn(self.ffn_norm(z)))
        return z


class LCRWrapperBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        ffn_expansion: int,
        cross_block_count: int,
        self_block_count: int,
        block_dropout: float = 0.0,
    ):
        super().__init__()
        block_dropout = _validate_dropout_prob("block_dropout", block_dropout)

        self.cross_blocks = nn.ModuleList(
            [
                AttnBlock(
                    dim=dim,
                    heads=heads,
                    ffn_expansion=ffn_expansion,
                    dropout=block_dropout,
                    use_context_norm=True,
                )
                for _ in range(cross_block_count)
            ]
        )
        self.self_blocks = nn.ModuleList(
            [
                AttnBlock(
                    dim=dim,
                    heads=heads,
                    ffn_expansion=ffn_expansion,
                    dropout=block_dropout,
                    exclude_self_value=True,
                )
                for _ in range(self_block_count)
            ]
        )

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        for cross_block in self.cross_blocks:
            z = cross_block(z, h)
        for self_block in self.self_blocks:
            z = self_block(z)
        return z


class ResDWBlock(nn.Module):
    def __init__(self, dim: int, ffn_expansion: int, dropout: float = 0.0):
        super().__init__()
        dropout = _validate_dropout_prob("dropout", dropout)
        self.conv_norm = RMSNorm2d(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
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


class EncoderSpatialBlock(nn.Module):
    def __init__(self, dim: int, ffn_expansion: int, dropout: float = 0.0):
        super().__init__()
        dropout = _validate_dropout_prob("dropout", dropout)
        self.spatial_norm = RMSNorm2d(dim)
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1)
        self.spatial_dropout = nn.Dropout(p=dropout)
        self.ffn_norm = RMSNorm2d(dim)
        self.ffn = DWConvFFN(dim=dim, expansion=ffn_expansion)
        self.ffn_dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

    def _spatial_residual(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial_norm(x)
        spatial = self.depthwise(spatial)
        spatial = self.pointwise(self.act(spatial))
        return self.spatial_dropout(spatial)

    def _ffn_residual(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn_dropout(self.ffn(self.ffn_norm(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._spatial_residual(x)
        x = x + self._ffn_residual(x)
        return x


class LatentEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dim: int,
        ffn_expansion: int,
        encoder_block_count: int,
        patch_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=1)
        self.patch_embed = nn.Conv2d(dim, dim, kernel_size=patch_size, stride=patch_size)
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderSpatialBlock(dim=dim, ffn_expansion=ffn_expansion, dropout=dropout)
                for _ in range(encoder_block_count)
            ]
        )
        self.act = nn.GELU()

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.proj(x))

    def _embed_patches(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.patch_embed(x))

    def _run_encoder_blocks(self, x: torch.Tensor) -> torch.Tensor:
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._project(x)
        x = self._embed_patches(x)
        return self._run_encoder_blocks(x)


class LatentDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        patch_size: int,
        ffn_expansion: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        dropout = _validate_dropout_prob("dropout", dropout)
        self.patch_size = patch_size
        self.full_dim = max(dim // 2, 1)
        self.block_latent = ResDWBlock(dim=dim, ffn_expansion=ffn_expansion, dropout=dropout)
        self.expand = nn.Conv2d(dim, self.full_dim * patch_size * patch_size, kernel_size=1)
        self.upsample = nn.PixelShuffle(patch_size)
        self.block_full = PointwiseResBlock(
            dim=self.full_dim,
            expansion=ffn_expansion,
            dropout=dropout,
        )
        self.out = nn.Conv2d(self.full_dim, self.full_dim, kernel_size=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        decoded = self.block_latent(latent)
        decoded = self.expand(decoded)
        decoded = self.upsample(decoded)
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
        ffn_expansion: int = 2,
        cross_block_count: int = 1,
        self_block_count: int = 1,
        encoder_block_count: int = 4,
        patch_size: int = 2,
        mask_bias_init: float = -2.0,
        block_dropout: float = 0.08,
        decoder_dropout: float = 0.05,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be greater than zero")
        if cross_block_count <= 0:
            raise ValueError("cross_block_count must be greater than zero")
        if self_block_count <= 0:
            raise ValueError("self_block_count must be greater than zero")
        if encoder_block_count <= 0:
            raise ValueError("encoder_block_count must be greater than zero")
        if patch_size <= 0:
            raise ValueError("patch_size must be greater than zero")
        block_dropout = _validate_dropout_prob("block_dropout", block_dropout)
        decoder_dropout = _validate_dropout_prob("decoder_dropout", decoder_dropout)

        self.sar_channels = sar_channels
        self.opt_channels = opt_channels
        self.dim = dim
        self.num_blocks = num_blocks
        self.cross_block_count = cross_block_count
        self.self_block_count = self_block_count
        self.encoder_block_count = encoder_block_count
        self.patch_size = patch_size
        self.block_dropout = block_dropout
        self.decoder_dropout = decoder_dropout

        self.sar_encoder = LatentEncoder(
            in_channels=sar_channels,
            dim=dim,
            ffn_expansion=ffn_expansion,
            encoder_block_count=encoder_block_count,
            patch_size=patch_size,
            dropout=block_dropout,
        )
        self.hsi_encoder = LatentEncoder(
            in_channels=opt_channels,
            dim=dim,
            ffn_expansion=ffn_expansion,
            encoder_block_count=encoder_block_count,
            patch_size=patch_size,
            dropout=block_dropout,
        )

        self.wrapper_blocks = nn.ModuleList(
            [
                LCRWrapperBlock(
                    dim=dim,
                    heads=heads,
                    ffn_expansion=ffn_expansion,
                    cross_block_count=cross_block_count,
                    self_block_count=self_block_count,
                    block_dropout=block_dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        self.candidate_decoder = LatentDecoder(
            dim=dim,
            patch_size=patch_size,
            ffn_expansion=ffn_expansion,
            dropout=decoder_dropout,
        )
        self.mask_decoder = LatentDecoder(
            dim=dim,
            patch_size=patch_size,
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

    def _validate_inputs(self, sar: torch.Tensor, cloudy: torch.Tensor) -> None:
        if sar.shape[0] != cloudy.shape[0] or sar.shape[-2:] != cloudy.shape[-2:]:
            raise ValueError("sar and cloudy inputs must share the same batch and spatial size")
        if sar.shape[1] != self.sar_channels:
            raise ValueError(f"expected sar to have {self.sar_channels} channels, got {sar.shape[1]}")
        if cloudy.shape[1] != self.opt_channels:
            raise ValueError(
                f"expected cloudy to have {self.opt_channels} channels, got {cloudy.shape[1]}"
            )

    def _run_wrapper_blocks(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        for wrapper_block in self.wrapper_blocks:
            z = wrapper_block(z, h)
        return z

    def _decode_candidate(self, z: torch.Tensor) -> torch.Tensor:
        candidate_features = self.candidate_decoder(z)
        return self.candidate_head(candidate_features)

    def _decode_mask(self, z: torch.Tensor) -> torch.Tensor:
        mask_features = self.mask_decoder(z)
        mask_logits = self.mask_out(mask_features)
        return torch.sigmoid(mask_logits)

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(sar, cloudy)

        z = self.sar_encoder(sar)
        h = self.hsi_encoder(cloudy)
        z = self._run_wrapper_blocks(z, h)

        candidate = self._decode_candidate(z)
        mask = self._decode_mask(z)
        return (1.0 - mask) * cloudy + mask * candidate
