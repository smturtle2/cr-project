"""Ablation: CrossModalBlock ×2 (body1+body2 뒤) — ×1 대비 성능 비교.

모델:  ACA_CRNet(use_cafm=False, use_cross_modal=True, cross_modal_num_blocks=2)
       - CrossModalBlock을 body1, body2 이후 2곳에 삽입
       - SARLightEncoder는 1개 (f_sar 공유)
Optim: AdamW(lr=1e-4)
Loss:  MAE (L1Loss)
Data:  train 512 / val 64 / test 64
Train: batch=4, epochs=20, crop 128×128 + flip + rot90 (Trainer 기본)
Baseline: artifacts/module1_mae_xmodal1_20ep (×1, 동일 조건)
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import main as shared_main
import torch
from torch import nn

from modules.metrics.metrics import MAE, PSNR, SSIM, SAM


_DEFAULT_BUILD_BEST_EPOCH_SELECTOR = shared_main.build_best_epoch_selector


def build_model() -> nn.Module:
    from modules.model.ACA_CRNet import ACA_CRNet

    return ACA_CRNet(
        use_cafm=False,
        use_cross_modal=True,
        cross_modal_heads=4,
        cross_modal_ffn_expansion=2.0,
        use_checkpoint=True,
        use_density_only=False,
        cross_modal_num_blocks=2,
    )


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=1e-4)


def build_loss() -> shared_main.LossFn:
    l1_loss = nn.L1Loss()

    def loss_fn(prediction: torch.Tensor, batch: shared_main.Batch) -> torch.Tensor:
        return l1_loss(prediction, batch["target"])

    return loss_fn


def build_metrics() -> dict[str, shared_main.MetricFn]:
    mods: dict[str, nn.Module] = {
        "mae": MAE(),
        "psnr": PSNR(),
        "ssim": SSIM(),
        "sam": SAM(),
    }

    def _wrap(mod: nn.Module) -> shared_main.MetricFn:
        def fn(prediction: torch.Tensor, batch: shared_main.Batch) -> torch.Tensor:
            return mod(prediction, batch["target"])
        return fn

    return {name: _wrap(m) for name, m in mods.items()}


def build_best_epoch_selector() -> shared_main.BestEpochSelector:
    return _DEFAULT_BUILD_BEST_EPOCH_SELECTOR()


@contextmanager
def use_local_builds() -> Iterator[None]:
    overrides = {
        "build_model": build_model,
        "build_optimizer": build_optimizer,
        "build_loss": build_loss,
        "build_metrics": build_metrics,
        "build_best_epoch_selector": build_best_epoch_selector,
    }
    originals = {name: getattr(shared_main, name) for name in overrides}

    for name, override in overrides.items():
        setattr(shared_main, name, override)

    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(shared_main, name, original)


def main() -> None:
    with use_local_builds():
        shared_main.main(
            batch_size=4,
            seed=42,
            max_epochs=20,
            train_max_samples=512,
            val_max_samples=64,
            test_max_samples=64,
            output_dir=Path("artifacts/module1_mae_xmodal2_20ep"),
            resume=None,
            run_test=True,
            num_examples=5,
            example_mode="best",
        )


if __name__ == "__main__":
    main()
