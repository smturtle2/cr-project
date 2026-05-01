# Density Gate Porting Source for `origin/main`

작성일: 2026-05-01

목적: 현재 `weight` 브랜치의 CAFM 실험 코드를 통째로 옮기지 않고, `origin/main`의 새 모델 구조에 `cosine`, `cosine_prior` density gate만 모듈화해서 장착하기 위한 소스 코드 정리.

기준 구조:

```text
origin/main
modules/model/baseline/ACA_CRNet.py
modules/model/module/base_module.py
modules/model/module/mask_module.py
modules/model/module/cross_attention_module.py
```

핵심 장착 방식:

```text
기존 main:
feature + cross_attention(sar, feature) * mask

density 장착 후:
feature + cross_attention(sar, feature) * mask * density
```

즉 density는 CAFM feature modulation 입력이 아니라, 새 main 모델의 SAR injection gate로 직접 사용한다.

---

## 1. 새 파일: `modules/model/density/base.py`

```python
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseDensityEstimator(nn.Module, ABC):
    """Common interface for pluggable density estimators."""

    @abstractmethod
    def forward(self, sar: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
        """Return a `(B, 1, H, W)` density map in `[0, 1]`."""
```

---

## 2. 새 파일: `modules/model/density/cosine.py`

```python
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseDensityEstimator


class CosineDensityEstimator(BaseDensityEstimator):
    """Feature cosine similarity followed by a small refiner."""

    def __init__(
        self,
        sar_channels: int = 2,
        optical_channels: int = 13,
        feat_dim: int = 32,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.sar_encoder = nn.Sequential(
            nn.Conv2d(sar_channels, feat_dim, kernel_size=1, bias=True),
            nn.GELU(),
        )
        self.opt_encoder = nn.Sequential(
            nn.Conv2d(optical_channels, feat_dim, kernel_size=1, bias=True),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(2 * feat_dim + 1, feat_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(feat_dim, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def encode(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sar_feat = self.sar_encoder(sar)
        opt_feat = self.opt_encoder(optical)
        similarity = F.cosine_similarity(sar_feat, opt_feat, dim=1, eps=1e-6).unsqueeze(1)
        return sar_feat, opt_feat, similarity

    def forward(self, sar: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
        sar_feat, opt_feat, similarity = self.encode(sar, optical)
        features = torch.cat([sar_feat, opt_feat, similarity], dim=1)
        return self.refine(features)
```

---

## 3. 새 파일: `modules/model/density/prior.py`

```python
from __future__ import annotations

import torch
from torch import nn


def _safe_rescale(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    scale = max(high - low, 1e-6)
    return ((x - low) / scale).clamp(0.0, 1.0)


def _normalized_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    denom = (a + b).clamp_min(1e-6)
    return (a - b) / denom


class OpticalRulePrior(nn.Module):
    """Sentinel-2 optical-only rule-based cloud-likeness prior."""

    def forward(self, optical: torch.Tensor) -> torch.Tensor:
        aerosol = optical[:, 0:1]
        blue = optical[:, 1:2]
        green = optical[:, 2:3]
        red = optical[:, 3:4]
        nir = optical[:, 7:8]
        cirrus = optical[:, 10:11]
        swir1 = optical[:, 11:12]

        score = torch.ones_like(blue)
        score = torch.minimum(score, _safe_rescale(blue, 0.1, 0.5))
        score = torch.minimum(score, _safe_rescale(aerosol, 0.1, 0.3))
        score = torch.minimum(score, _safe_rescale(aerosol + cirrus, 0.4, 0.9))
        score = torch.minimum(score, _safe_rescale(red + green + blue, 0.2, 0.8))

        ndmi = _normalized_difference(nir, swir1)
        score = torch.minimum(score, _safe_rescale(ndmi, -0.1, 0.1))

        ndsi = _normalized_difference(green, swir1)
        score = torch.minimum(score, _safe_rescale(ndsi, 0.8, 0.6))

        score = torch.nn.functional.max_pool2d(score, kernel_size=5, stride=1, padding=2)
        score = torch.nn.functional.avg_pool2d(score, kernel_size=7, stride=1, padding=3)
        return score.clamp(0.0, 1.0)
```

---

## 4. 새 파일: `modules/model/density/cosine_prior.py`

```python
from __future__ import annotations

import torch

from .base import BaseDensityEstimator
from .cosine import CosineDensityEstimator
from .prior import OpticalRulePrior


class CosinePriorDensityEstimator(BaseDensityEstimator):
    """Blend learned cosine density with a fixed optical-only prior."""

    def __init__(
        self,
        sar_channels: int = 2,
        optical_channels: int = 13,
        feat_dim: int = 32,
        prior_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.cosine = CosineDensityEstimator(
            sar_channels=sar_channels,
            optical_channels=optical_channels,
            feat_dim=feat_dim,
        )
        self.prior = OpticalRulePrior()
        self.prior_weight = float(prior_weight)

    def forward(self, sar: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
        cosine_density = self.cosine(sar, optical)
        prior_density = self.prior(optical)
        weight = self.prior_weight
        return ((1.0 - weight) * cosine_density + weight * prior_density).clamp(0.0, 1.0)
```

---

## 5. 새 파일: `modules/model/density/factory.py`

```python
from __future__ import annotations

from .base import BaseDensityEstimator
from .cosine import CosineDensityEstimator
from .cosine_prior import CosinePriorDensityEstimator
from .prior import OpticalRulePrior


def build_density_estimator(
    mode: str,
    *,
    sar_channels: int = 2,
    optical_channels: int = 13,
    feat_dim: int = 32,
    prior_weight: float = 0.5,
) -> BaseDensityEstimator:
    if mode == "cosine":
        return CosineDensityEstimator(
            sar_channels=sar_channels,
            optical_channels=optical_channels,
            feat_dim=feat_dim,
        )
    if mode == "cosine_prior":
        return CosinePriorDensityEstimator(
            sar_channels=sar_channels,
            optical_channels=optical_channels,
            feat_dim=feat_dim,
            prior_weight=prior_weight,
        )
    if mode == "prior":
        return OpticalRulePrior()
    raise ValueError(f"unsupported density mode: {mode}")
```

---

## 6. 새 파일: `modules/model/density/__init__.py`

```python
from .base import BaseDensityEstimator
from .cosine import CosineDensityEstimator
from .cosine_prior import CosinePriorDensityEstimator
from .factory import build_density_estimator
from .prior import OpticalRulePrior

__all__ = [
    "BaseDensityEstimator",
    "CosineDensityEstimator",
    "CosinePriorDensityEstimator",
    "OpticalRulePrior",
    "build_density_estimator",
]
```

---

## 7. 수정 파일: `modules/model/module/base_module.py`

`origin/main`의 기존 파일은 다음 형태다.

```python
class BaseModule(nn.Module):
    ...
    def forward(self, sar, cloudy, feature):
        mask = self.mask(sar, cloudy)
        out = self.cross_attn(sar, feature)
        return feature + out * mask
```

아래처럼 `density` 인자를 추가한다.

```python
# -*- coding: utf-8 -*-

import torch.nn as nn

from .cross_attention_module import CrossAttentionModule
from .mask_module import MaskModule


class BaseModule(nn.Module):
    def __init__(
        self,
        sar_channels,
        cloudy_channels,
        feature_channels,
        num_heads=4,
        patch_size=2,
        self_num_layers=2,
        cross_num_layers=2,
    ):
        super(BaseModule, self).__init__()
        self.mask = MaskModule(
            sar_channels,
            cloudy_channels,
            feature_channels,
            num_heads=num_heads,
            patch_size=patch_size,
            num_layers=self_num_layers,
        )
        self.cross_attn = CrossAttentionModule(
            sar_channels,
            feature_channels,
            num_heads,
            patch_size=patch_size,
            self_num_layers=self_num_layers,
            cross_num_layers=cross_num_layers,
        )

    def forward(self, sar, cloudy, feature, density=None):
        mask = self.mask(sar, cloudy)
        out = self.cross_attn(sar, feature)

        gate = mask
        if density is not None:
            gate = gate * density

        return feature + out * gate
```

주의:

```text
mask shape:    (B, feature_channels, H, W)
density shape: (B, 1, H, W)
```

PyTorch broadcasting으로 `mask * density`가 채널 방향에 자동 확장된다.

---

## 8. 수정 파일: `modules/model/baseline/ACA_CRNet.py`

`origin/main`의 `ACA_CRNet`에 density 옵션만 추가한다.

### 8-1. import 추가

파일 상단에 추가:

```python
from ..density import build_density_estimator
```

### 8-2. `__init__` 인자 추가

기존:

```python
class ACA_CRNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        alpha=0.1,
        num_layers=16,
        feature_sizes=256,
        gpu_ids=[],
        ca=DefaultConAttn,
        ca_kwargs=None,
        is_baseline=False
    ):
```

수정:

```python
class ACA_CRNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        alpha=0.1,
        num_layers=16,
        feature_sizes=256,
        gpu_ids=[],
        ca=DefaultConAttn,
        ca_kwargs=None,
        is_baseline=False,
        density_mode=None,
        density_feat_dim=32,
        density_prior_weight=0.5,
    ):
```

### 8-3. density estimator 속성 추가

`sar_channels = in_channels - out_channels` 바로 뒤에 추가:

```python
self.density_mode = density_mode
self.last_density = None
self.last_cloudy = None

if density_mode is not None:
    self.density_estimator = build_density_estimator(
        density_mode,
        sar_channels=sar_channels,
        optical_channels=out_channels,
        feat_dim=density_feat_dim,
        prior_weight=density_prior_weight,
    )
else:
    self.density_estimator = None
```

### 8-4. `forward` 수정

기존:

```python
def forward(self, sar, cloudy):
    x = torch.cat((sar, cloudy), dim=1)
    out = x
    for layer in self.net:
        if isinstance(layer, BaseModule):
            out = layer(sar, cloudy, out)
        else:
            out = layer(out)
    return cloudy + out
```

수정:

```python
def forward(self, sar, cloudy):
    self.last_cloudy = cloudy
    if self.density_estimator is not None:
        self.last_density = self.density_estimator(sar, cloudy)
    else:
        self.last_density = None

    x = torch.cat((sar, cloudy), dim=1)
    out = x
    for layer in self.net:
        if isinstance(layer, BaseModule):
            out = layer(sar, cloudy, out, density=self.last_density)
        else:
            out = layer(out)
    return cloudy + out
```

---

## 9. `ACA_CRNet.py` 전체 참고본

아래는 `origin/main` 구조에 density 변경을 반영한 전체 참고본이다. 실제 포팅 시에는 기존 파일에 필요한 diff만 적용해도 된다.

```python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn import init

from .ca import ConAttn
from ..density import build_density_estimator
from ..module.base_module import BaseModule

DefaultConAttn = ConAttn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=256, alpha=0.1):
        super(ResBlock, self).__init__()
        m = OrderedDict()
        m["conv1"] = nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False, stride=1, padding=1)
        m["relu1"] = nn.ReLU(True)
        m["conv2"] = nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, stride=1, padding=1)
        self.net = nn.Sequential(m)
        self.relu = nn.Sequential(nn.ReLU(True))
        self.alpha = alpha

    def forward(self, x):
        out = self.net(x)
        out = self.alpha * out + x
        return out


class ResBlock_att(nn.Module):
    def __init__(self, in_channels, out_channels=256, alpha=0.1, ca=DefaultConAttn, ca_kwargs=None):
        super(ResBlock_att, self).__init__()
        ca_kwargs = {} if ca_kwargs is None else dict(ca_kwargs)
        m = OrderedDict()
        m["conv1"] = nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False, stride=2, padding=1)
        m["relu1"] = nn.ReLU(True)
        m["conv2"] = nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, stride=1, padding=1)
        m["relu2"] = nn.ReLU(True)
        m["att"] = ca(input_channels=out_channels, output_channels=out_channels, ksize=1, stride=1, **ca_kwargs)
        self.net = nn.Sequential(m)
        self.alpha = alpha

    def forward(self, x):
        out = self.net(x)
        out = torch.nn.functional.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)
        out = self.alpha * out + x
        return out


def init_net(net, init_type="kaiming-uniform", gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def init_weights(net, init_type="kaiming-uniform", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "kaiming-uniform":
                init.kaiming_uniform_(m.weight.data, mode="fan_in", nonlinearity="relu")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(f"initialization method [{init_type}] is not implemented")
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print(f"initialize network with {init_type}")
    net.apply(init_func)


class ACA_CRNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        alpha=0.1,
        num_layers=16,
        feature_sizes=256,
        gpu_ids=[],
        ca=DefaultConAttn,
        ca_kwargs=None,
        is_baseline=False,
        density_mode=None,
        density_feat_dim=32,
        density_prior_weight=0.5,
    ):
        super(ACA_CRNet, self).__init__()
        ca_kwargs = {} if ca_kwargs is None else dict(ca_kwargs)
        sar_channels = in_channels - out_channels

        self.density_mode = density_mode
        self.last_density = None
        self.last_cloudy = None
        if density_mode is not None:
            self.density_estimator = build_density_estimator(
                density_mode,
                sar_channels=sar_channels,
                optical_channels=out_channels,
                feat_dim=density_feat_dim,
                prior_weight=density_prior_weight,
            )
        else:
            self.density_estimator = None

        m = []
        m.append(nn.Conv2d(in_channels, out_channels=feature_sizes, kernel_size=3, bias=True, stride=1, padding=1))
        m.append(nn.ReLU(True))
        for i in range(num_layers):
            if i == num_layers // 2:
                m.append(ResBlock_att(feature_sizes, feature_sizes, alpha, ca=ca, ca_kwargs=ca_kwargs))
                if not is_baseline:
                    m.append(BaseModule(sar_channels, out_channels, feature_sizes))
            elif i == num_layers * 3 // 4:
                m.append(ResBlock_att(feature_sizes, feature_sizes, alpha, ca=ca, ca_kwargs=ca_kwargs))
                if not is_baseline:
                    m.append(BaseModule(sar_channels, out_channels, feature_sizes))
            else:
                m.append(ResBlock(feature_sizes, feature_sizes, alpha))

        m.append(nn.Conv2d(feature_sizes, out_channels, kernel_size=3, bias=True, stride=1, padding=1))
        self.net = nn.ModuleList(m)
        self.gpu_ids = gpu_ids
        if len(self.gpu_ids) > 0:
            assert torch.cuda.is_available()
            self.net.to(self.gpu_ids[0])
        if is_baseline:
            init_weights(self.net, "kaiming-uniform")

    def forward(self, sar, cloudy):
        self.last_cloudy = cloudy
        if self.density_estimator is not None:
            self.last_density = self.density_estimator(sar, cloudy)
        else:
            self.last_density = None

        x = torch.cat((sar, cloudy), dim=1)
        out = x
        for layer in self.net:
            if isinstance(layer, BaseModule):
                out = layer(sar, cloudy, out, density=self.last_density)
            else:
                out = layer(out)
        return cloudy + out
```

---

## 10. 새 파일: `modules/metrics/density_eval.py`

```python
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
```

---

## 11. 새 파일: `scripts/eval_density.py`

`main.py`의 helper 함수 이름은 새 브랜치 구현 상태에 따라 다를 수 있다. 아래 스크립트는 `origin/main`의 `seed_everything`, `normalize_rgb_triplet`, `normalize_map`이 유지된다는 가정이다.

`ACA_CRNet` import 경로는 main 기준으로 `modules.model.baseline.ACA_CRNet`을 사용한다.

```python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main as shared_main
from cr_train.data import DATASET_ID, build_dataloader
from cr_train.data.dataset import prepare_split
from cr_train.data.runtime import ensure_split_cache
from modules.metrics.density_eval import compute_band_proxies, summarize_density
from modules.model.baseline.ACA_CRNet import ACA_CRNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Density estimator offline evaluation")
    parser.add_argument(
        "--density-modes",
        nargs="+",
        choices=("cosine", "cosine_prior", "prior"),
        default=("cosine", "cosine_prior"),
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Optional mapping in the form mode=/path/to/checkpoint.pt",
    )
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "density_compare")
    return parser.parse_args()


def build_loader(*, split: str, max_samples: int, batch_size: int, seed: int):
    cache_root = Path("artifacts") / "density_eval_cache"
    ensure_split_cache(
        split=split,
        dataset_name=DATASET_ID,
        revision=None,
        max_samples=max_samples,
        seed=seed,
        cache_root=cache_root,
    )
    prepared = prepare_split(
        split=split,
        dataset_name=DATASET_ID,
        revision=None,
        max_samples=max_samples,
        seed=seed,
        epoch=0,
        training=False,
        cache_root=cache_root,
    )
    return build_dataloader(
        prepared,
        batch_size=batch_size,
        num_workers=0,
        training=False,
        seed=seed,
        epoch=0,
        include_metadata=False,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        drop_last=False,
    )


def parse_checkpoint_map(entries: list[str]) -> dict[str, Path]:
    checkpoint_map: dict[str, Path] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"invalid checkpoint mapping: {entry}")
        mode, raw_path = entry.split("=", 1)
        checkpoint_map[mode] = Path(raw_path)
    return checkpoint_map


def save_summary_plot(results: dict[str, dict[str, float]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    modes = list(results.keys())
    metric_specs = [
        ("corr_weighted", "Corr Weighted", False),
        ("clear_mean_d", "Clear Mean D", True),
        ("thick_mean_d", "Thick Mean D", False),
        ("top10_proxy", "Top10 Proxy", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (key, title, lower_is_better) in zip(axes.flat, metric_specs):
        values = [results[mode][key] for mode in modes]
        colors = ["#3b82f6", "#f59e0b", "#10b981"]
        ax.bar(modes, values, color=colors[: len(modes)])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        best_value = min(values) if lower_is_better else max(values)
        for index, value in enumerate(values):
            mark = " *" if value == best_value else ""
            ax.text(index, value, f"{value:.3f}{mark}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_example_figure(
    *,
    mode: str,
    sample_index: int,
    density: torch.Tensor,
    cloudy: torch.Tensor,
    target: torch.Tensor,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    proxy = compute_band_proxies(cloudy.unsqueeze(0), target.unsqueeze(0))["weighted"][0, 0]
    cloudy_rgb, _, target_rgb = shared_main.normalize_rgb_triplet(cloudy, target, target)
    density_map = shared_main.normalize_map(density[0])
    proxy_map = shared_main.normalize_map(proxy)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    panels = (
        ("Cloudy RGB", cloudy_rgb, None),
        ("Target RGB", target_rgb, None),
        ("Weighted Proxy", proxy_map, "magma"),
        ("Density", density_map, "viridis"),
    )
    for ax, (title, image, cmap) in zip(axes, panels):
        if cmap is None:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(f"{mode} example {sample_index}", fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(top=0.83)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_mode(
    *,
    mode: str,
    checkpoint_path: Path | None,
    split: str,
    max_samples: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    num_examples: int,
    output_dir: Path,
) -> dict[str, object]:
    model = ACA_CRNet(
        in_channels=15,
        out_channels=13,
        density_mode=mode,
        is_baseline=False,
    ).to(device)
    model.eval()

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)

    loader = build_loader(
        split=split,
        max_samples=max_samples,
        batch_size=batch_size,
        seed=seed,
    )

    metrics = []
    saved_examples = 0
    with torch.no_grad():
        for batch in loader:
            sar = batch["sar"].to(device)
            cloudy = batch["cloudy"]
            target = batch["target"]
            density = model.density_estimator(sar, cloudy.to(device)).cpu()
            metrics.append(summarize_density(density, cloudy, target))

            for index in range(density.shape[0]):
                if saved_examples >= num_examples:
                    break
                save_example_figure(
                    mode=mode,
                    sample_index=saved_examples + 1,
                    density=density[index],
                    cloudy=cloudy[index],
                    target=target[index],
                    output_path=output_dir / mode / f"example_{saved_examples + 1:02d}.png",
                )
                saved_examples += 1

    aggregate = {}
    for key in asdict(metrics[0]).keys():
        aggregate[key] = sum(getattr(item, key) for item in metrics) / len(metrics)
    return aggregate


def main() -> None:
    args = parse_args()
    shared_main.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_map = parse_checkpoint_map(args.checkpoint)
    output_dir = args.output_dir / args.split / f"samples_{args.max_samples}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, float]] = {}
    for mode in args.density_modes:
        results[mode] = evaluate_mode(
            mode=mode,
            checkpoint_path=checkpoint_map.get(mode),
            split=args.split,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
            num_examples=args.num_examples,
            output_dir=output_dir / "examples",
        )

    payload = {
        "density_modes": list(args.density_modes),
        "split": args.split,
        "max_samples": args.max_samples,
        "results": results,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    (output_dir / "metrics.json").write_text(text + "\n", encoding="utf-8")
    save_summary_plot(results, output_dir / "summary.png")


if __name__ == "__main__":
    main()
```

---

## 12. 테스트 참고: `tests/test_density_estimators.py`

새 브랜치에서는 import 경로를 `modules.model.density`와 `modules.model.baseline.ACA_CRNet` 기준으로 둔다.

```python
from __future__ import annotations

import unittest

import torch

from modules.metrics.density_eval import summarize_density
from modules.model.baseline.ACA_CRNet import ACA_CRNet
from modules.model.density import (
    CosineDensityEstimator,
    CosinePriorDensityEstimator,
    OpticalRulePrior,
)


class DensityEstimatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sar = torch.rand(2, 2, 16, 16)
        self.cloudy = torch.rand(2, 13, 16, 16)
        self.target = torch.rand(2, 13, 16, 16)

    def test_estimators_return_density_map_in_unit_interval(self) -> None:
        estimators = [
            CosineDensityEstimator(feat_dim=8),
            CosinePriorDensityEstimator(feat_dim=8),
            OpticalRulePrior(),
        ]

        for estimator in estimators:
            with self.subTest(estimator=type(estimator).__name__):
                density = estimator(self.sar, self.cloudy)
                self.assertEqual(density.shape, (2, 1, 16, 16))
                self.assertGreaterEqual(float(density.min()), 0.0)
                self.assertLessEqual(float(density.max()), 1.0)

    def test_aca_crnet_wires_density_mode_without_changing_interface(self) -> None:
        model = ACA_CRNet(
            in_channels=15,
            out_channels=13,
            num_layers=4,
            feature_sizes=16,
            density_feat_dim=8,
            density_mode="cosine_prior",
            is_baseline=False,
        )
        prediction = model(self.sar, self.cloudy)
        self.assertEqual(prediction.shape, self.cloudy.shape)
        self.assertIsNotNone(model.last_density)
        self.assertEqual(model.last_density.shape, (2, 1, 16, 16))

    def test_density_summary_orders_clear_thin_thick(self) -> None:
        weighted = (self.cloudy - self.target).abs().mean(dim=1, keepdim=True)
        result = summarize_density(weighted, self.cloudy, self.target)
        self.assertGreaterEqual(result.thick_mean_d, result.thin_mean_d)
        self.assertGreaterEqual(result.thin_mean_d, result.clear_mean_d)


if __name__ == "__main__":
    unittest.main()
```

---

## 13. `main.py` 연결 참고

`origin/main`의 `main.py`는 `build_model()`이 placeholder다. 새 브랜치에서 CLI를 붙일 때 필요한 최소 인자는 다음이다.

```python
parser.add_argument(
    "--density-mode",
    choices=("none", "cosine", "cosine_prior", "prior"),
    default="none",
)
parser.add_argument("--density-feat-dim", type=int, default=32)
parser.add_argument("--density-prior-weight", type=float, default=0.5)
```

`build_model()`에서는:

```python
def build_model() -> nn.Module:
    density_mode = None if _args.density_mode == "none" else _args.density_mode
    return ACA_CRNet(
        in_channels=15,
        out_channels=13,
        density_mode=density_mode,
        density_feat_dim=_args.density_feat_dim,
        density_prior_weight=_args.density_prior_weight,
        is_baseline=False,
    )
```

실험 명령 예:

```bash
python main.py \
  --density-mode cosine \
  --output-dir artifacts/main_density/cosine_5ep \
  --batch-size 2 \
  --max-epochs 5 \
  --save-every 1
```

```bash
python main.py \
  --density-mode cosine_prior \
  --output-dir artifacts/main_density/cosine_prior_5ep \
  --batch-size 2 \
  --max-epochs 5 \
  --save-every 1
```

---

## 14. 포팅 시 가져오지 말 것

현재 `weight` 브랜치에서 아래 파일/구조는 새 main 브랜치로 가져오지 않는다.

```text
modules/model/cafm/ACA_CRNet.py
modules/model/cafm/cafm.py
tmp_main.py 전체
best.pt
history.png
artifacts/
examples/
```

이유:

```text
새 main 모델은 이미 BaseModule 기반 SAR cross-attention gate 구조를 갖고 있다.
기존 CAFM feature modulation 코드를 섞으면 원인 분석이 흐려진다.
```

---

## 15. 분석용 비교 실험

최소 비교군:

```text
A. main baseline: density_mode=None
B. main + cosine
C. main + cosine_prior
```

가능하면 추가:

```text
D. main + prior
```

해석:

```text
cosine만 collapse하면 cosine estimator saturation 문제 가능성
cosine_prior만 collapse하면 prior 결합 또는 prior scale 문제 가능성
둘 다 안정되면 기존 CAFM modulation 구조가 문제였을 가능성
둘 다 무너지면 reconstruction objective와 density gate 해석이 충돌할 가능성
```

