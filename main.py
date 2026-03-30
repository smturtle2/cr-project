from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cr_train import MAE, Trainer, TrainerConfig, build_sen12mscr_loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal cr-train entry point.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=("official", "seeded_scene"), default="official")
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--train-max-batches", type=int, default=None)
    parser.add_argument("--val-max-batches", type=int, default=None)
    parser.add_argument("--test-max-batches", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--run-test", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model() -> nn.Module:
    # Trainer will call model(sar, cloudy), so build your model to accept two tensors.
    raise NotImplementedError("Replace build_model() with your project model.")


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    # Keep this aligned with build_model().
    raise NotImplementedError("Replace build_optimizer() with your optimizer.")


def build_criterion() -> nn.Module:
    # Replace this if you want a different training loss.
    return nn.L1Loss()


def build_metrics() -> list[nn.Module]:
    # Add or replace metrics here.
    return [MAE()]


def flatten_record(record: dict) -> dict[str, int | float]:
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

    try:
        import matplotlib.pyplot as plt
    except ImportError:
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


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # cr-train already knows how to build the streaming SEN12MS-CR loaders.
    train_loader, val_loader, test_loader = build_sen12mscr_loaders(
        batch_size=args.batch_size,
        seed=args.seed,
        split=args.split,
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
            train_max_batches=args.train_max_batches,
            val_max_batches=args.val_max_batches,
            test_max_batches=args.test_max_batches,
            checkpoint_dir=args.checkpoint_dir,
        ),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    # Resume first, then keep iterating the epoch records from Trainer.step().
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

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
        print(format_row({"epoch": trainer.state.epoch, "global_step": trainer.state.global_step, **test_row}))


if __name__ == "__main__":
    main()
