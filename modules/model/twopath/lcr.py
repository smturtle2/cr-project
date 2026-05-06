from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import init

from ..lcr.model import LCRWrapperBlock, LatentDecoder, LatentEncoder


class LCR(nn.Module):
    def __init__(
        self,
        *,
        sar_channels: int,
        opt_channels: int,
        dim: int = 64,
        num_blocks: int = 6,
        heads: int = 4,
        ffn_expansion: int = 2,
        cross_block_count: int = 1,
        self_block_count: int = 1,
        encoder_block_count: int = 4,
        patch_size: int = 2,
        mask_init_prob: float = 0.05,
        block_dropout: float = 0.08,
        decoder_dropout: float = 0.05,
    ):
        super().__init__()
        if sar_channels <= 0:
            raise ValueError("sar_channels must be greater than zero")
        if opt_channels <= 0:
            raise ValueError("opt_channels must be greater than zero")
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
        if not 0.0 < mask_init_prob < 1.0:
            raise ValueError("mask_init_prob must be between 0 and 1")

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
        self.cloudy_encoder = LatentEncoder(
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
        self.mask_decoder = LatentDecoder(
            dim=dim,
            patch_size=patch_size,
            ffn_expansion=ffn_expansion,
            dropout=decoder_dropout,
        )
        self.mask_head = nn.Conv2d(self.mask_decoder.full_dim, opt_channels, kernel_size=1)

        init.constant_(self.mask_head.weight, 0.0)
        init.constant_(self.mask_head.bias, math.log(mask_init_prob / (1.0 - mask_init_prob)))

    def _validate_inputs(self, sar: torch.Tensor, cloudy: torch.Tensor) -> None:
        if sar.shape[0] != cloudy.shape[0] or sar.shape[-2:] != cloudy.shape[-2:]:
            raise ValueError("sar and cloudy inputs must share the same batch and spatial size")
        if sar.shape[1] != self.sar_channels:
            raise ValueError(f"expected sar to have {self.sar_channels} channels, got {sar.shape[1]}")
        if cloudy.shape[1] != self.opt_channels:
            raise ValueError(
                f"expected cloudy to have {self.opt_channels} channels, got {cloudy.shape[1]}"
            )
        height, width = sar.shape[-2:]
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("sar and cloudy spatial size must be divisible by patch_size")

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(sar, cloudy)

        z = self.sar_encoder(sar)
        h = self.cloudy_encoder(cloudy)
        for wrapper_block in self.wrapper_blocks:
            z = wrapper_block(z, h)

        mask_features = self.mask_decoder(z)
        return self.mask_head(mask_features)
