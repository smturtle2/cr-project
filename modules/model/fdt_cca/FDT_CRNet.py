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
        feature_extractor_layers=1,
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
        del cca_layers
        if out_channels != cloudy_channels:
            raise ValueError("out_channels must match cloudy_channels")

        super().__init__()
        self.return_decomposition = return_decomposition
        self.fdt = FDT_CCA(
            sar_channels=sar_channels,
            cloudy_channels=cloudy_channels,
            dim=dim,
            num_heads=num_heads,
            num_layers=fdt_layers,
            extractor_dims=extractor_dims,
            feature_extractor_layers=feature_extractor_layers,
        )
        self.cloud_channels = self.fdt.extractor_dims[0]
        crnet = CCA_CRNet(
            out_channels=out_channels,
            alpha=alpha,
            num_layers=cr_layers,
            feature_sizes=self.fdt.dim,
            cloud_channels=self.cloud_channels,
            ca=ca,
            ca_kwargs=ca_kwargs,
        )
        self.crnet = init_net(crnet, init_type=init_type)
        self.crnet.cca.reset_parameters()

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor):
        (
            fdt_feature,
            sar_feat,
            cld_clear,
            cld_cloud,
        ) = self.fdt(sar, cloudy)
        prediction, candidate, mask = self.crnet(
            fdt_feature,
            cld_cloud,
            cloudy,
            return_candidate=True,
            return_mask=True,
        )
        if self.return_decomposition:
            return prediction, candidate, mask, sar_feat, cld_clear, cld_cloud
        return prediction
