from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


KEY_BAND_INDICES = {
    "blue": 1,
    "red": 3,
    "nir": 7,
    "cirrus": 10,
    "swir1": 11,
}


@dataclass(slots=True)
class GateEvalResult:
    corr_blue: float
    corr_red: float
    corr_nir: float
    corr_cirrus: float
    corr_swir1: float
    corr_weighted: float
    clear_mean_g: float
    thin_mean_g: float
    thick_mean_g: float
    top10_proxy: float
    top20_proxy: float


def prepare_gate_for_eval(gate: torch.Tensor, target_hw: tuple[int, int] | None = None) -> torch.Tensor:
    """Convert mask/density gates to a comparable (B, 1, H, W) map in [0, 1]."""
    if gate.ndim != 4:
        raise ValueError(f"gate must be a 4D tensor, got shape {tuple(gate.shape)}")
    if gate.shape[1] != 1:
        gate = gate.mean(dim=1, keepdim=True)
    if target_hw is not None and gate.shape[-2:] != target_hw:
        gate = F.interpolate(gate, size=target_hw, mode="bilinear", align_corners=False)
    return gate.clamp(0.0, 1.0)


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


def summarize_gate(
    gate: torch.Tensor,
    cloudy: torch.Tensor,
    target: torch.Tensor,
) -> GateEvalResult:
    gate = prepare_gate_for_eval(gate, target_hw=cloudy.shape[-2:])
    proxies = compute_band_proxies(cloudy, target)
    weighted = proxies["weighted"]

    flat_proxy = weighted.reshape(-1)
    q30 = torch.quantile(flat_proxy, 0.30)
    q70 = torch.quantile(flat_proxy, 0.70)
    clear_mask = weighted <= q30
    thick_mask = weighted >= q70
    thin_mask = (~clear_mask) & (~thick_mask)

    flat_gate = gate.reshape(-1)
    gate_sorted, _ = torch.sort(flat_gate, descending=True)
    top10_count = max(1, math.ceil(flat_gate.numel() * 0.10))
    top20_count = max(1, math.ceil(flat_gate.numel() * 0.20))
    top10_threshold = gate_sorted[top10_count - 1]
    top20_threshold = gate_sorted[top20_count - 1]
    top10_mask = gate >= top10_threshold
    top20_mask = gate >= top20_threshold

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        if not torch.any(mask):
            return 0.0
        return float(values[mask].float().mean())

    return GateEvalResult(
        corr_blue=_safe_corr(gate, proxies["blue"]),
        corr_red=_safe_corr(gate, proxies["red"]),
        corr_nir=_safe_corr(gate, proxies["nir"]),
        corr_cirrus=_safe_corr(gate, proxies["cirrus"]),
        corr_swir1=_safe_corr(gate, proxies["swir1"]),
        corr_weighted=_safe_corr(gate, weighted),
        clear_mean_g=masked_mean(gate, clear_mask),
        thin_mean_g=masked_mean(gate, thin_mask),
        thick_mean_g=masked_mean(gate, thick_mask),
        top10_proxy=masked_mean(weighted, top10_mask),
        top20_proxy=masked_mean(weighted, top20_mask),
    )
