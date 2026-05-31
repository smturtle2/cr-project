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

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor):
        fdt_output = self.fdt(sar, cloudy)
        cr_output = self.crnet(
            fdt_output["feature"],
            fdt_output["cloud_feat"],
            cloudy,
            return_candidate=True,
            return_mask=True,
        )
        prediction = cr_output["prediction"]
        if self.return_decomposition:
            return {
                "prediction": prediction,
                "candidate": cr_output["candidate"],
                "mask": cr_output["mask"],
                "sar_feat": fdt_output["sar_feat"],
                "clear_feat": fdt_output["clear_feat"],
                "cloud_feat": fdt_output["cloud_feat"],
            }
        return prediction
