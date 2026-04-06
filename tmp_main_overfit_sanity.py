"""Sanity check: overfitting on a tiny training set.

목적: 모델이 학습 데이터를 완벽히 외울 수 있는지 확인.
      train example의 Prediction이 Target과 동일해지면 모델 용량/학습이 정상.

모델:  ACA_CRNet(use_cafm=False, use_cross_modal=True, cross_modal_num_blocks=2)
Optim: AdamW(lr=1e-4)
Loss:  MAE (L1Loss)
Data:  train 16 / val 16 / test 16
Train: batch=4, epochs=100, crop 128×128 + flip + rot90
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import main as shared_main
import torch
from torch import nn

from modules.metrics.metrics import MAE, PSNR, SSIM, SAM


TRAIN_MAX_SAMPLES = 16

_DEFAULT_BUILD_BEST_EPOCH_SELECTOR = shared_main.build_best_epoch_selector
_ORIGINAL_MAYBE_UPDATE = shared_main.maybe_update_best_state


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


def maybe_update_best_state_with_train(trainer, record, *, output_dir, device,
                                        best_state, selector, example_mode,
                                        val_max_samples, num_examples):
    new_best = _ORIGINAL_MAYBE_UPDATE(
        trainer, record, output_dir=output_dir, device=device,
        best_state=best_state, selector=selector,
        example_mode=example_mode, val_max_samples=val_max_samples,
        num_examples=num_examples,
    )
    # best epoch가 갱신되었을 때만 train example도 저장
    if new_best is not best_state and new_best is not None:
        epoch = int(record["epoch"])
        shared_main.save_examples_for_epoch(
            trainer, device=device, split="train",
            max_samples=TRAIN_MAX_SAMPLES,
            stage="train", epoch=epoch,
            output_dir=output_dir / "examples" / "best" / f"epoch_{epoch:03d}",
            num_examples=num_examples,
        )
    return new_best


@contextmanager
def use_local_builds() -> Iterator[None]:
    overrides = {
        "build_model": build_model,
        "build_optimizer": build_optimizer,
        "build_loss": build_loss,
        "build_metrics": build_metrics,
        "build_best_epoch_selector": build_best_epoch_selector,
        "maybe_update_best_state": maybe_update_best_state_with_train,
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
            max_epochs=100,
            train_max_samples=TRAIN_MAX_SAMPLES,
            val_max_samples=16,
            test_max_samples=16,
            output_dir=Path("artifacts/sanity_overfit"),
            resume=None,
            run_test=True,
            num_examples=5,
            example_mode="best",
        )


if __name__ == "__main__":
    main()
