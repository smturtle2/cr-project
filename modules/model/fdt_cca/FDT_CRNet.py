from __future__ import annotations

import torch
import torch.nn as nn

from ..fdt.ACA_CRNet import init_net
from ..fdt.FDT_CRNet import DefaultConAttn
from .cca_crnet import CCA_CRNet
from .fdt import FDTCCA


class FDT_CRNet_CCA(nn.Module):
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
            raise ValueError("out_channels must match cloudy_channels")

        super().__init__()
        self.return_decomposition = return_decomposition
        self.fdt = FDTCCA(
            sar_channels=sar_channels,
            cloudy_channels=cloudy_channels,
            dim=dim,
            num_heads=num_heads,
            num_layers=fdt_layers,
        )
        crnet = CCA_CRNet(
            out_channels=out_channels,
            alpha=alpha,
            num_layers=cr_layers,
            feature_sizes=dim,
            comp_channels=dim // 2,
            ca=ca,
            ca_kwargs=ca_kwargs,
        )
        self.crnet = init_net(crnet, init_type=init_type)
        self.crnet.zero_init_cca()

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor):
        (
            fdt_feature,
            sar_com,
            cld_com,
            sar_comp,
            cld_comp,
        ) = self.fdt(sar, cloudy)
        prediction = self.crnet(fdt_feature, cld_comp)
        if self.return_decomposition:
            return prediction, sar_com, cld_com, sar_comp, cld_comp
        return prediction
