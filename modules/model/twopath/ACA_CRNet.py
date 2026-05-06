# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
from torch.nn import init

from ..baseline.ACA_CRNet import ACA_CRNet
from ..baseline.ca_flash import ConAttn as FlashConAttn
from ..module.base_module import BaseModule


class ACA_CRNetRaw(ACA_CRNet):
    def forward(self, sar, cloudy):
        x = torch.cat((sar, cloudy), dim=1)
        out = x
        for layer in self.net:
            if isinstance(layer, BaseModule):
                out = layer(sar, cloudy, out)
            else:
                out = layer(out)
        return out


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
    ):
        super(TwoPathACA_CRNet, self).__init__()
        if not 0.0 < mask_init_prob < 1.0:
            raise ValueError("mask_init_prob must be between 0 and 1")
        ca_kwargs = {} if ca_kwargs is None else dict(ca_kwargs)
        self.mask_net = ACA_CRNetRaw(
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
        mask_head = self.mask_net.net[-1]
        init.constant_(mask_head.weight, 0.0)
        init.constant_(mask_head.bias, math.log(mask_init_prob / (1.0 - mask_init_prob)))
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

    def forward(self, sar, cloudy):
        mask = torch.sigmoid(self.mask_net(sar, cloudy))
        candidate = self.candidate_net(sar, cloudy)
        return cloudy * (1 - mask) + candidate * mask
