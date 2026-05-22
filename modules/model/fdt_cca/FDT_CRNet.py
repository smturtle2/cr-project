from __future__ import annotations

import torch
import torch.nn as nn

from ..fdt.ACA_CRNet import init_net
from ..fdt.FDT_CRNet import DefaultConAttn
from .cca_crnet import CCA_CRNet
from .fdt import FDT_CCA


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
        extractor_dims=(128, 256, 512),
        cca_layers=None,
    ):
        if out_channels != cloudy_channels:
            raise ValueError("out_channels must match cloudy_channels")

        super().__init__()
        self.return_decomposition = return_decomposition
        if cca_layers is None:
            cca_layers = fdt_layers
        self.fdt = FDT_CCA(
            sar_channels=sar_channels,
            cloudy_channels=cloudy_channels,
            dim=dim,
            num_heads=num_heads,
            num_layers=fdt_layers,
            extractor_dims=extractor_dims,
        )
        self.component_channels = self.fdt.extractor_dims[0]
        crnet = CCA_CRNet(
            out_channels=out_channels,
            alpha=alpha,
            num_layers=cr_layers,
            feature_sizes=self.fdt.dim,
            comp_channels=self.component_channels,
            cca_layers=cca_layers,
            num_heads=num_heads,
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
        prediction = cloudy + self.crnet(fdt_feature, cld_comp)
        if self.return_decomposition:
            return prediction, sar_com, cld_com, sar_comp, cld_comp
        return prediction
