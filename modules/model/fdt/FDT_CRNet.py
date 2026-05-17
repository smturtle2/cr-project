from __future__ import annotations

import torch
import torch.nn as nn

from .ACA_CRNet import ACA_CRNet, BaselineInput, DefaultConAttn, init_net
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
            ca=ca,
            ca_kwargs=ca_kwargs,
            mode="direct",
        )
        self.crnet = init_net(crnet, init_type=init_type)

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor):
        fdt_output = self.fdt(sar, cloudy)
        prediction = cloudy + self.crnet(fdt_output.feature)
        if self.return_decomposition:
            return prediction, fdt_output
        return prediction


class FDT_CRNet_Side(nn.Module):
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
        side_mode="residual",
        return_decomposition=False,
    ):
        super().__init__()
        self.return_decomposition = return_decomposition
        self.input = BaselineInput(sar_channels + cloudy_channels, dim)
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
            ca=ca,
            ca_kwargs=ca_kwargs,
            mode="side",
            side_mode=side_mode,
            num_heads=num_heads,
        )
        self.input = init_net(self.input, init_type=init_type)
        self.crnet = init_net(crnet, init_type=init_type)

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor):
        main_feature = self.input(sar, cloudy)
        fdt_output = self.fdt(sar, cloudy)
        prediction = cloudy + self.crnet(main_feature, fdt_output.feature)
        if self.return_decomposition:
            return prediction, fdt_output
        return prediction
