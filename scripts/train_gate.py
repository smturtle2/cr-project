from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main as shared_main
from modules.metrics import PSNR, SAM, SSIM
from modules.model.baseline import ca as ca_base
from modules.model.baseline import ca_flash, ca_optim
from modules.model.baseline.ACA_CRNet import ACA_CRNet


_args: argparse.Namespace | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACA-CRNet gate comparison training")
    parser.add_argument("--gate-mode", choices=("mask", "cosine", "cosine_prior"), default="mask")
    parser.add_argument("--gate-feat-dim", type=int, default=32)
    parser.add_argument("--gate-prior-weight", type=float, default=0.5)
    parser.add_argument("--ca-mode", choices=("base", "optim", "flash"), default="base")
    parser.add_argument("--ca-chunk-size", type=int, default=512, help="chunk size used only with --ca-mode optim")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--train-max-samples", type=int, default=4096)
    parser.add_argument("--val-max-samples", type=int, default=512)
    parser.add_argument("--test-max-samples", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "gate_runs")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--num-workers", default="auto")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--run-test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-examples", type=int, default=4)
    return parser.parse_args()


def build_model() -> nn.Module:
    assert _args is not None
    ca_cls = {
        "base": ca_base.ConAttn,
        "optim": ca_optim.ConAttn,
        "flash": ca_flash.ConAttn,
    }[_args.ca_mode]
    ca_kwargs = {"chunk_size": _args.ca_chunk_size} if _args.ca_mode == "optim" else None
    model = ACA_CRNet(
        in_channels=15,
        out_channels=13,
        ca=ca_cls,
        ca_kwargs=ca_kwargs,
        gate_mode=_args.gate_mode,
        gate_feat_dim=_args.gate_feat_dim,
        gate_prior_weight=_args.gate_prior_weight,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"gate_mode: {_args.gate_mode} | "
        f"ca_mode: {_args.ca_mode} | "
        f"gate_feat_dim: {_args.gate_feat_dim} | "
        f"prior_weight: {_args.gate_prior_weight} | "
        f"params: {total_params:,}"
    )
    return model


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    assert _args is not None
    return torch.optim.AdamW(model.parameters(), lr=_args.lr)


def build_loss() -> shared_main.LossFn:
    criterion = nn.L1Loss()

    def loss_fn(prediction: torch.Tensor, batch: shared_main.Batch) -> torch.Tensor:
        return criterion(prediction, batch["target"])

    return loss_fn


def build_metrics() -> dict[str, shared_main.MetricFn]:
    metric_modules = {"psnr": PSNR(), "ssim": SSIM(), "sam": SAM()}

    def wrap(metric: nn.Module) -> shared_main.MetricFn:
        def metric_fn(prediction: torch.Tensor, batch: shared_main.Batch) -> torch.Tensor:
            return metric(prediction, batch["target"])

        return metric_fn

    return {name: wrap(metric) for name, metric in metric_modules.items()}


@contextmanager
def use_local_builds() -> Iterator[None]:
    overrides = {
        "build_model": build_model,
        "build_optimizer": build_optimizer,
        "build_loss": build_loss,
        "build_metrics": build_metrics,
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
    global _args
    _args = parse_args()
    output_dir = Path(_args.output_dir) / _args.gate_mode
    with use_local_builds():
        shared_main.main(
            batch_size=_args.batch_size,
            accum_steps=_args.accum_steps,
            seed=_args.seed,
            max_epochs=_args.max_epochs,
            train_max_samples=_args.train_max_samples,
            val_max_samples=_args.val_max_samples,
            test_max_samples=_args.test_max_samples,
            output_dir=output_dir,
            cache_dir=_args.cache_dir,
            resume=_args.resume,
            num_workers=_args.num_workers,
            save_every_n_epochs=_args.save_every,
            run_test=_args.run_test,
            num_examples=_args.num_examples,
            example_splits=["test"],
        )


if __name__ == "__main__":
    main()
