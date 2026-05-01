from __future__ import annotations

import math
from dataclasses import dataclass

import torch


KEY_BAND_INDICES = {
    "blue": 1,
    "red": 3,
    "nir": 7,
    "cirrus": 10,
    "swir1": 11,
}


@dataclass(slots=True)
class DensityEvalResult:
    corr_blue: float
    corr_red: float
    corr_nir: float
    corr_cirrus: float
    corr_swir1: float
    corr_weighted: float
    clear_mean_d: float
    thin_mean_d: float
    thick_mean_d: float
    top10_proxy: float
    top20_proxy: float


def _safe_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.sqrt((a.square().sum() * b.square().sum()).clamp_min(1e-12))
    if denom.item() == 0:
        return 0.0
    return float((a * b).sum() / denom)


def compute_band_proxies(cloudy: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    proxies = {
        name: (cloudy[:, index:index + 1] - target[:, index:index + 1]).abs()
        for name, index in KEY_BAND_INDICES.items()
    }
    proxies["weighted"] = (
        0.30 * proxies["blue"]
        + 0.30 * proxies["cirrus"]
        + 0.20 * proxies["swir1"]
        + 0.10 * proxies["red"]
        + 0.10 * proxies["nir"]
    )
    return proxies


def summarize_density(
    density: torch.Tensor,
    cloudy: torch.Tensor,
    target: torch.Tensor,
) -> DensityEvalResult:
    proxies = compute_band_proxies(cloudy, target)
    weighted = proxies["weighted"]

    flat_proxy = weighted.reshape(-1)
    q30 = torch.quantile(flat_proxy, 0.30)
    q70 = torch.quantile(flat_proxy, 0.70)
    clear_mask = weighted <= q30
    thick_mask = weighted >= q70
    thin_mask = (~clear_mask) & (~thick_mask)

    flat_density = density.reshape(-1)
    density_sorted, _ = torch.sort(flat_density, descending=True)
    top10_count = max(1, math.ceil(flat_density.numel() * 0.10))
    top20_count = max(1, math.ceil(flat_density.numel() * 0.20))
    top10_threshold = density_sorted[top10_count - 1]
    top20_threshold = density_sorted[top20_count - 1]
    top10_mask = density >= top10_threshold
    top20_mask = density >= top20_threshold

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        if not torch.any(mask):
            return 0.0
        return float(values[mask].float().mean())

    return DensityEvalResult(
        corr_blue=_safe_corr(density, proxies["blue"]),
        corr_red=_safe_corr(density, proxies["red"]),
        corr_nir=_safe_corr(density, proxies["nir"]),
        corr_cirrus=_safe_corr(density, proxies["cirrus"]),
        corr_swir1=_safe_corr(density, proxies["swir1"]),
        corr_weighted=_safe_corr(density, weighted),
        clear_mean_d=masked_mean(density, clear_mask),
        thin_mean_d=masked_mean(density, thin_mask),
        thick_mean_d=masked_mean(density, thick_mask),
        top10_proxy=masked_mean(weighted, top10_mask),
        top20_proxy=masked_mean(weighted, top20_mask),
    )

