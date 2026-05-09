from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseGateEstimator


def compute_prior_refine_components_v2(optical: torch.Tensor) -> dict[str, torch.Tensor]:
    """Extract cloud/shadow priors from optical bands (Sentinel-2 like).

    Args:
        optical: (B, 13, H, W) tensor with bands:
            0:blue, 1:green, 2:red, 3:rededge_1, 4:rededge_2, 5:rededge_3,
            6:nir, 7:narrow_nir, 8:swir1, 9:swir2, 10:cirrus, 11:aerosol

    Returns:
        dict with keys: nonclear_prior, cloud_prior, shadow_prior, aerosol,
                       blue, green, red, nir, cirrus, swir1
    """
    b, _, h, w = optical.shape

    # Extract channels
    blue = optical[:, 0:1]      # B2
    green = optical[:, 1:2]     # B3
    red = optical[:, 2:3]       # B4
    nir = optical[:, 6:7]       # B8
    swir1 = optical[:, 8:9]     # B11
    cirrus = optical[:, 10:11]  # B9
    aerosol = optical[:, 11:12] # Aerosol

    # Normalize to [0, 1]
    blue_n = blue.clamp(0, 1)
    green_n = green.clamp(0, 1)
    red_n = red.clamp(0, 1)
    nir_n = nir.clamp(0, 1)
    swir1_n = swir1.clamp(0, 1)

    ndvi = (nir_n - red_n) / (nir_n + red_n + 1e-6)
    ndvi_01 = ((ndvi + 1.0) * 0.5).clamp(0.0, 1.0)
    vegetation = torch.sigmoid((ndvi - 0.25) / 0.08)

    # Cloud detection (brightness + whiteness), with vegetation suppression.
    brightness = (blue_n + green_n + red_n) / 3.0
    whiteness = 1.0 - torch.std(torch.cat([blue_n, green_n, red_n], dim=1), dim=1, keepdim=True)
    cloud_prior = brightness * whiteness * (1.0 - 0.15 * vegetation)

    # Shadow detection: dark optical pixels with suppressed NIR/SWIR.
    # This avoids opening clear dark vegetation as shadow.
    darkness = 1.0 - brightness
    low_nir_swir = (0.6 * (1.0 - nir_n) + 0.4 * (1.0 - swir1_n)).clamp(0.0, 1.0)
    shadow_prior = darkness * low_nir_swir * (1.0 - 0.20 * ndvi_01)

    # Nonclear (max of cloud and shadow)
    nonclear_prior = torch.maximum(cloud_prior, shadow_prior)

    return {
        "nonclear_prior": nonclear_prior,
        "cloud_prior": cloud_prior,
        "shadow_prior": shadow_prior,
        "aerosol": aerosol,
        "blue": blue,
        "green": green,
        "red": red,
        "nir": nir,
        "cirrus": cirrus,
        "swir1": swir1,
    }


def sharpen_map(x: torch.Tensor, *, threshold: float = 0.35, temperature: float = 0.08) -> torch.Tensor:
    """Convert a soft evidence map into a stronger 0..1 open/closed signal."""
    return torch.sigmoid((x.clamp(0.0, 1.0) - threshold) / temperature)


class PriorRefineGateEstimatorV4(BaseGateEstimator):
    """Denoise and correct a heuristic nonclear prior.

    The estimator keeps the heuristic prior as the anchor, but learns two
    corrections:
      remove: suppress noisy prior detections
      add:    recover damaged regions missed by the prior

    gate = prior_open * (1 - remove) + (1 - prior_open) * add
    """

    def __init__(
        self,
        sar_channels: int = 2,
        optical_channels: int = 13,
        feat_dim: int = 32,
        cloud_threshold: float = 0.30,
        shadow_threshold: float = 0.20,
        prior_temperature: float = 0.08,
        clear_guard_threshold: float = 0.05,
        clear_guard_temperature: float = 0.03,
    ) -> None:
        super().__init__()
        del sar_channels
        if optical_channels < 12:
            raise ValueError("PriorRefineGateEstimatorV4 expects optical_channels >= 12")
        self.cloud_threshold = float(cloud_threshold)
        self.shadow_threshold = float(shadow_threshold)
        self.prior_temperature = float(prior_temperature)
        self.clear_guard_threshold = float(clear_guard_threshold)
        self.clear_guard_temperature = float(clear_guard_temperature)

        input_channels = 4 + 7  # prior_open, nonclear, cloud, shadow + spectral bands
        hidden = max(8, feat_dim)
        self.trunk = nn.Sequential(
            nn.Conv2d(input_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.remove_head = nn.Sequential(nn.Conv2d(hidden, 1, kernel_size=1), nn.Sigmoid())
        self.add_head = nn.Sequential(nn.Conv2d(hidden, 1, kernel_size=1), nn.Sigmoid())
        nn.init.constant_(self.remove_head[0].bias, -4.0)
        nn.init.constant_(self.add_head[0].bias, -5.0)

        self.last_prior_open: torch.Tensor | None = None
        self.last_cloud_open: torch.Tensor | None = None
        self.last_shadow_open: torch.Tensor | None = None
        self.last_clear_guard: torch.Tensor | None = None
        self.last_nonclear_prior: torch.Tensor | None = None
        self.last_cloud_prior: torch.Tensor | None = None
        self.last_shadow_prior: torch.Tensor | None = None
        self.last_remove: torch.Tensor | None = None
        self.last_add: torch.Tensor | None = None

    def forward(self, sar: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
        del sar
        comps = compute_prior_refine_components_v2(optical)
        cloud_open = sharpen_map(
            comps["cloud_prior"],
            threshold=self.cloud_threshold,
            temperature=self.prior_temperature,
        )
        shadow_open = sharpen_map(
            comps["shadow_prior"],
            threshold=self.shadow_threshold,
            temperature=self.prior_temperature,
        )
        prior_open = torch.maximum(cloud_open, shadow_open)
        nonclear_confidence = torch.maximum(comps["cloud_prior"], comps["shadow_prior"])
        clear_guard = sharpen_map(
            nonclear_confidence,
            threshold=self.clear_guard_threshold,
            temperature=self.clear_guard_temperature,
        )

        features = torch.cat(
            [
                prior_open,
                comps["nonclear_prior"],
                comps["cloud_prior"],
                comps["shadow_prior"],
                comps["aerosol"],
                comps["blue"],
                comps["green"],
                comps["red"],
                comps["nir"],
                comps["cirrus"],
                comps["swir1"],
            ],
            dim=1,
        )
        hidden = self.trunk(features)
        remove = self.remove_head(hidden)
        add = self.add_head(hidden)
        gate = prior_open * (1.0 - remove) + (1.0 - prior_open) * add * clear_guard
        gate = torch.maximum(gate, cloud_open * 0.77)
        gate = torch.maximum(gate, shadow_open * 0.70)

        self.last_prior_open = prior_open
        self.last_cloud_open = cloud_open
        self.last_shadow_open = shadow_open
        self.last_clear_guard = clear_guard
        self.last_nonclear_prior = comps["nonclear_prior"].detach()
        self.last_cloud_prior = comps["cloud_prior"].detach()
        self.last_shadow_prior = comps["shadow_prior"].detach()
        self.last_remove = remove
        self.last_add = add
        return gate.clamp(0.0, 1.0)
