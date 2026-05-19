from __future__ import annotations

import torch

from ..fdt.FDT_CRNet import DefaultConAttn, FDT_CRNet_Direct
from .fdt import FDTMask, MaskEncoder


class FDT_CRNet_Mask(FDT_CRNet_Direct):
    def __init__(
        self,
        sar_channels=2,
        cloudy_channels=13,
        out_channels=13,
        dim=256,
        fdt_layers=2,
        cr_layers=16,
        num_heads=4,
        alpha=0.1,
        init_type="kaiming-uniform",
        ca=DefaultConAttn,
        ca_kwargs=None,
        return_decomposition=False,
    ):
        if out_channels != cloudy_channels:
            raise ValueError("out_channels must match cloudy_channels for mask blending")

        super().__init__(
            sar_channels=sar_channels,
            cloudy_channels=cloudy_channels,
            out_channels=out_channels,
            dim=dim,
            fdt_layers=fdt_layers,
            cr_layers=cr_layers,
            num_heads=num_heads,
            alpha=alpha,
            init_type=init_type,
            ca=ca,
            ca_kwargs=ca_kwargs,
            return_decomposition=return_decomposition,
        )
        self.fdt = FDTMask(
            sar_channels=sar_channels,
            cloudy_channels=cloudy_channels,
            dim=dim,
            num_layers=fdt_layers,
            num_heads=num_heads,
        )
        self.mask_encoder = MaskEncoder(
            dim=dim,
            out_channels=out_channels,
            num_layers=fdt_layers,
            heads=num_heads,
        )

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor):
        (
            fdt_feature,
            sar_com,
            cld_com,
            sar_comp,
            cld_comp,
            cld_comp_l,
        ) = self.fdt(sar, cloudy)
        candidate = self.crnet(fdt_feature)
        mask = self.mask_encoder(cld_comp_l)
        prediction = (1.0 - mask) * cloudy + mask * candidate
        if self.return_decomposition:
            return prediction, sar_com, cld_com, sar_comp, cld_comp
        return prediction
