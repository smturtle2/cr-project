from __future__ import annotations

import torch
from torch import nn

from ..baseline.ca_flash import ConAttn as FlashConAttn
from .ACA_CRNet import ACA_CRNetRaw
from .lcr import LCR


class TwoPathACA_CRNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        alpha=0.1,
        num_layers=16,
        feature_sizes=256,
        gpu_ids=[],
        ca_kwargs=None,
        mask_init_prob=0.05,
        lcr_dim=128,
        lcr_num_blocks=6,
        lcr_heads=4,
        lcr_ffn_expansion=2,
        lcr_cross_block_count=1,
        lcr_self_block_count=1,
        lcr_encoder_block_count=4,
        lcr_patch_size=2,
        lcr_block_dropout=0.08,
        lcr_decoder_dropout=0.05,
    ):
        super().__init__()
        sar_channels = in_channels - out_channels
        self.mask_net = LCR(
            sar_channels=sar_channels,
            opt_channels=out_channels,
            dim=lcr_dim,
            num_blocks=lcr_num_blocks,
            heads=lcr_heads,
            ffn_expansion=lcr_ffn_expansion,
            cross_block_count=lcr_cross_block_count,
            self_block_count=lcr_self_block_count,
            encoder_block_count=lcr_encoder_block_count,
            patch_size=lcr_patch_size,
            mask_init_prob=mask_init_prob,
            block_dropout=lcr_block_dropout,
            decoder_dropout=lcr_decoder_dropout,
        )
        ca_kwargs = {} if ca_kwargs is None else dict(ca_kwargs)
        self.candidate_net = ACA_CRNetRaw(
            in_channels,
            out_channels,
            alpha=alpha,
            num_layers=num_layers,
            feature_sizes=feature_sizes,
            gpu_ids=gpu_ids,
            ca=FlashConAttn,
            ca_kwargs=ca_kwargs,
            is_baseline=True,
        )

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor) -> torch.Tensor:
        mask = torch.sigmoid(self.mask_net(sar, cloudy))
        candidate = self.candidate_net(sar, cloudy)
        return cloudy * (1.0 - mask) + candidate * mask
