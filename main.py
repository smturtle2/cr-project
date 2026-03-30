from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from cr_train import MAE, Trainer, TrainerConfig, build_sen12mscr_loaders


RGB_CHANNELS = (3, 2, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SEN12MS-CR 학습 엔트리 포인트")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=("official", "seeded_scene"), default="official")
    parser.add_argument("--io-profile", choices=("smooth", "conservative"), default="smooth")
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--train-max-samples", type=int, default=2048)
    parser.add_argument("--val-max-samples", type=int, default=512)
    parser.add_argument("--test-max-samples", type=int, default=512)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--run-test", action="store_true")
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--example-stage", choices=("val", "test"), default="val")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    # 실험 재현성을 위해 파이썬, 넘파이, 파이토치 시드를 함께 고정한다.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model() -> nn.Module:
    # Trainer는 model(sar, cloudy)를 호출하므로 입력 시그니처를 맞춰 구현한다.
    raise NotImplementedError("프로젝트 모델로 build_model()을 교체하세요.")


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    # build_model()에서 만든 파라미터와 함께 사용할 optimizer를 정의한다.
    raise NotImplementedError("프로젝트 optimizer로 build_optimizer()를 교체하세요.")


def build_criterion() -> nn.Module:
    # 기본 복원 손실은 L1로 둔다.
    return nn.L1Loss()


def build_metrics() -> list[nn.Module]:
    # Trainer가 epoch 평균을 집계할 metric 목록이다.
    return [MAE()]


def flatten_record(record: dict) -> dict[str, int | float]:
    # Trainer가 stage별로 나눠 주는 로그를 한 줄짜리 dict로 평탄화한다.
    row: dict[str, int | float] = {
        "epoch": int(record["epoch"]),
        "global_step": int(record["global_step"]),
    }
    for stage in ("train", "val", "test"):
        for name, value in record.get(stage, {}).items():
            row[f"{stage}_{name}"] = float(value)
    return row


def format_value(value: int | float | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def format_row(row: dict[str, int | float]) -> str:
    metric_keys = sorted(key for key in row if key not in {"epoch", "global_step"})
    parts = [f"epoch={row['epoch']}", f"global_step={row['global_step']}"]
    parts.extend(f"{key}={format_value(row[key])}" for key in metric_keys)
    return " | ".join(parts)


def print_history(history: list[dict[str, int | float]]) -> None:
    if not history:
        return

    keys = ["epoch", "global_step"] + sorted(
        {key for row in history for key in row if key not in {"epoch", "global_step"}}
    )
    widths = {
        key: max(len(key), *(len(format_value(row.get(key))) for row in history))
        for key in keys
    }

    print("\nhistory")
    print(" ".join(key.ljust(widths[key]) for key in keys))
    for row in history:
        print(" ".join(format_value(row.get(key)).ljust(widths[key]) for key in keys))

    for best_key in ("val_loss", "train_loss"):
        values = [row for row in history if best_key in row]
        if values:
            best_row = min(values, key=lambda row: float(row[best_key]))
            print(f"\nbest {best_key}: {format_row(best_row)}")
            break


def save_history_plot(history: list[dict[str, int | float]], path: Path) -> None:
    if not history:
        return

    metric_keys = sorted(
        {key for row in history for key in row if key not in {"epoch", "global_step"}}
    )
    loss_keys = [key for key in metric_keys if key.endswith("_loss")]
    other_keys = [key for key in metric_keys if key not in loss_keys]
    groups = [keys for keys in (loss_keys, other_keys) if keys]
    if not groups:
        return

    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(len(groups), 1, figsize=(10, 4 * len(groups)), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, keys in zip(axes, groups):
        for key in keys:
            values = [row.get(key, np.nan) for row in history]
            ax.plot(epochs, values, marker="o", linewidth=2.2, markersize=5, label=key)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("value")

    axes[0].set_title("training history")
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved plot: {path}")


def select_rgb(image: torch.Tensor) -> torch.Tensor:
    # 13채널 Sentinel-2 계열은 일반적으로 B4, B3, B2를 RGB로 본다.
    return image[list(RGB_CHANNELS)]


def normalize_rgb_triplet(
    cloudy: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb_tensors = [select_rgb(tensor).detach().cpu().float() for tensor in (cloudy, prediction, target)]
    merged = np.concatenate(
        [tensor.permute(1, 2, 0).reshape(-1, 3).numpy() for tensor in rgb_tensors],
        axis=0,
    )
    low = np.quantile(merged, 0.02, axis=0)
    high = np.quantile(merged, 0.98, axis=0)
    scale = np.maximum(high - low, 1e-6)

    images: list[np.ndarray] = []
    for tensor in rgb_tensors:
        image = tensor.permute(1, 2, 0).numpy()
        image = np.clip((image - low) / scale, 0.0, 1.0)
        images.append(image ** (1.0 / 2.2))
    return images[0], images[1], images[2]


def normalize_map(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().float().numpy()
    low, high = np.quantile(array, [0.02, 0.98])
    return np.clip((array - low) / max(high - low, 1e-6), 0.0, 1.0)


def save_restoration_examples(
    model: nn.Module,
    dataloader,
    device: torch.device,
    output_dir: Path,
    num_examples: int,
    stage: str,
) -> list[Path]:
    # 학습 후 cloudy / prediction / target / SAR / error를 한 장에 묶어 저장한다.
    if num_examples <= 0:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()

    saved_paths: list[Path] = []
    iterator = iter(dataloader)
    try:
        with torch.no_grad():
            while len(saved_paths) < num_examples:
                try:
                    batch = next(iterator)
                except StopIteration:
                    break

                sar, cloudy = batch["inputs"]
                target = batch["target"]
                metadata = batch["metadata"]
                prediction = model(sar.to(device), cloudy.to(device)).cpu()

                for index in range(prediction.shape[0]):
                    cloudy_rgb, prediction_rgb, target_rgb = normalize_rgb_triplet(
                        cloudy[index],
                        prediction[index],
                        target[index],
                    )
                    sar_map = normalize_map(sar[index].mean(dim=0))
                    error_map = normalize_map((prediction[index] - target[index]).abs().mean(dim=0))

                    title = (
                        f"{stage} example {len(saved_paths) + 1} | "
                        f"{metadata['season'][index]}/scene_{metadata['scene'][index]}/patch_{metadata['patch'][index]}"
                    )

                    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
                    fig.suptitle(title, fontsize=11)
                    panels = (
                        ("Cloudy RGB", cloudy_rgb, None),
                        ("Prediction RGB", prediction_rgb, None),
                        ("Target RGB", target_rgb, None),
                        ("SAR Mean", sar_map, "gray"),
                        ("Abs Error", error_map, "magma"),
                    )

                    for ax, (panel_title, image, cmap) in zip(axes, panels):
                        if cmap is None:
                            ax.imshow(image)
                        else:
                            ax.imshow(image, cmap=cmap)
                        ax.set_title(panel_title)
                        ax.axis("off")

                    fig.tight_layout()
                    fig.subplots_adjust(top=0.80)
                    path = output_dir / f"{stage}_example_{len(saved_paths) + 1:02d}.png"
                    fig.savefig(path, dpi=180, bbox_inches="tight")
                    plt.close(fig)
                    saved_paths.append(path)

                    if len(saved_paths) == num_examples:
                        break
    finally:
        # 스트리밍 iterator를 조기 종료하는 경우 worker/prefetch 상태를 바로 정리한다.
        del iterator
        model.train(was_training)

    print(f"\nsaved examples: {output_dir}")
    return saved_paths


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 최신 cr-train은 file-shard 스트리밍 + cache preset 조합을 io_profile로 제어한다.
    train_loader, val_loader, test_loader = build_sen12mscr_loaders(
        batch_size=args.batch_size,
        seed=args.seed,
        split=args.split,
        io_profile=args.io_profile,
    )

    model = build_model().to(device)
    optimizer = build_optimizer(model)
    criterion = build_criterion()
    metrics = build_metrics()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        metrics=metrics,
        config=TrainerConfig(
            max_epochs=args.max_epochs,
            train_max_samples=args.train_max_samples,
            val_max_samples=args.val_max_samples,
            test_max_samples=args.test_max_samples,
            checkpoint_dir=args.checkpoint_dir,
        ),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    # epoch별 로그를 모아 표와 그래프로 다시 본다.
    history: list[dict[str, int | float]] = []
    for record in trainer.step():
        row = flatten_record(record)
        history.append(row)
        print(format_row(row))

    print_history(history)
    save_history_plot(history, args.checkpoint_dir / "history.png")

    if args.run_test:
        test_row = {f"test_{name}": float(value) for name, value in trainer.test().items()}
        print("\ntest")
        print(
            format_row(
                {
                    "epoch": trainer.state.epoch,
                    "global_step": trainer.state.global_step,
                    **test_row,
                }
            )
        )

    example_loader = test_loader if args.example_stage == "test" else val_loader
    save_restoration_examples(
        model=model,
        dataloader=example_loader,
        device=device,
        output_dir=args.checkpoint_dir / "examples",
        num_examples=args.num_examples,
        stage=args.example_stage,
    )


if __name__ == "__main__":
    main()
