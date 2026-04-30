# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.nn as nn


class MaskModule(nn.Module):
    def __init__(self, sar_channels, cloudy_channels, feature_channels):
        super(MaskModule, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(sar_channels + cloudy_channels, feature_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, sar, cloudy, feature):
        if sar.shape[-2:] != feature.shape[-2:]:
            sar = F.interpolate(sar, size=feature.shape[-2:], mode="bilinear", align_corners=True)
        if cloudy.shape[-2:] != feature.shape[-2:]:
            cloudy = F.interpolate(cloudy, size=feature.shape[-2:], mode="bilinear", align_corners=True)
        return self.net(torch.cat((sar, cloudy), dim=1))
