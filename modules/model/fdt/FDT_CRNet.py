from __future__ import annotations

import torch
import torch.nn as nn

from .ACA_CRNet import ACA_CRNet, DefaultConAttn, init_net
from .fdt import FDT


class FDT_CRNet_Direct(nn.Module):
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
        gpu_ids=[],
        ca=DefaultConAttn,
        ca_kwargs=None,
        return_decomposition=False,
    ):
        super().__init__()
        self.return_decomposition = return_decomposition
        self.fdt = FDT(
            sar_channels=sar_channels,
            cloudy_channels=cloudy_channels,
            dim=dim,
            num_layers=fdt_layers,
            num_heads=num_heads,
        )
        crnet = ACA_CRNet(
            out_channels=out_channels,
            alpha=alpha,
            num_layers=cr_layers,
            feature_sizes=dim,
            gpu_ids=[],
            ca=ca,
            ca_kwargs=ca_kwargs,
        )
        self.crnet = init_net(crnet, init_type=init_type, gpu_ids=gpu_ids)

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor):
        fdt_feature, sar_com, cld_com, sar_comp, cld_comp = self.fdt(sar, cloudy)
        prediction = cloudy + self.crnet(fdt_feature)
        if self.return_decomposition:
            return prediction, sar_com, cld_com, sar_comp, cld_comp
        return prediction
