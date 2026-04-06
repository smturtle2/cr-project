"""Sanity check: overfitting on a tiny training set.

목적: 모델이 학습 데이터를 완벽히 외울 수 있는지 확인.
      train example의 Prediction이 Target과 동일해지면 모델 용량/학습이 정상.

모델:  ACA_CRNet(use_cafm=False, use_cross_modal=True, cross_modal_num_blocks=2)
Optim: AdamW(lr=1e-4)
Loss:  MAE (L1Loss)
Data:  train 16 / val 16 / test 16
Train: batch=4, epochs=100, crop 128×128 + flip + rot90

학습 전 train dataset 전체(16장) 저장 + 매 epoch마다 train/val example 1장 저장 + history.png 갱신.
"""

from __future__ import annotations

from pathlib import Path

import main as shared_main
import torch
from torch import nn

from modules.metrics.metrics import MAE, PSNR, SSIM, SAM


TRAIN_MAX_SAMPLES = 16
VAL_MAX_SAMPLES = 16
TEST_MAX_SAMPLES = 16
SAVE_EVERY = 1  # 매 epoch마다 example 저장 + plot 갱신
NUM_EXAMPLES = 1
OUTPUT_DIR = Path("artifacts/sanity_overfit")


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


def save_examples(trainer, device, epoch):
    """Train + val example을 함께 저장."""
    example_dir = OUTPUT_DIR / "examples" / f"epoch_{epoch:03d}"
    for split, stage, max_samples in [
        ("train", "train", TRAIN_MAX_SAMPLES),
        ("validation", "val", VAL_MAX_SAMPLES),
    ]:
        shared_main.save_examples_for_epoch(
            trainer, device=device, split=split,
            max_samples=max_samples, stage=stage, epoch=epoch,
            output_dir=example_dir, num_examples=NUM_EXAMPLES,
        )


def main() -> None:
    shared_main.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared_main.print_hf_auth_status()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model().to(device)
    optimizer = build_optimizer(model)
    trainer = shared_main.Trainer(
        model=model,
        optimizer=optimizer,
        loss=build_loss(),
        metrics=build_metrics(),
        max_train_samples=TRAIN_MAX_SAMPLES,
        max_val_samples=VAL_MAX_SAMPLES,
        max_test_samples=TEST_MAX_SAMPLES,
        output_dir=OUTPUT_DIR,
        batch_size=4,
        epochs=100,
        seed=42,
        train_crop_size=128,
        train_random_flip=True,
        train_random_rot90=True,
    )

    # 학습 전: train dataset 전체를 초기 모델(랜덤 가중치)로 저장
    shared_main.save_examples_for_epoch(
        trainer, device=device, split="train",
        max_samples=TRAIN_MAX_SAMPLES, stage="train",
        epoch=0,
        output_dir=OUTPUT_DIR / "examples" / "dataset",
        num_examples=TRAIN_MAX_SAMPLES,
    )

    history: list[dict[str, int | float]] = []

    while trainer.current_epoch < trainer.epochs:
        record = trainer.step()
        shared_main.append_history(history, record, global_step=trainer.global_step)
        epoch = int(record["epoch"])

        # 매 SAVE_EVERY epoch마다 또는 마지막 epoch에 example 저장 + plot 갱신
        if epoch % SAVE_EVERY == 0 or epoch == trainer.epochs:
            save_examples(trainer, device, epoch)
            shared_main.save_history_plot(history, OUTPUT_DIR / "history.png")

    # 최종 plot
    shared_main.save_history_plot(history, OUTPUT_DIR / "history.png")


if __name__ == "__main__":
    main()
