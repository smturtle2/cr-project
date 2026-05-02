from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main as shared_main
from modules.metrics.gate_eval import compute_band_proxies, prepare_gate_for_eval, summarize_gate
from modules.model.baseline import ca as ca_base
from modules.model.baseline import ca_flash, ca_optim
from modules.model.baseline.ACA_CRNet import ACA_CRNet


class IdentityCA(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAR injection gate offline evaluation")
    parser.add_argument(
        "--gate-modes",
        nargs="+",
        choices=("mask", "cosine", "cosine_prior", "dafi_diff"),
        default=("mask", "cosine", "cosine_prior", "dafi_diff"),
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", action="append", default=[], help="Optional mapping in the form mode=/path/to/checkpoint.pt")
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "gate_compare")
    parser.add_argument("--gate-feat-dim", type=int, default=32)
    parser.add_argument("--gate-prior-weight", type=float, default=0.5)
    parser.add_argument("--ca-mode", choices=("base", "optim", "flash"), default="base")
    parser.add_argument("--ca-chunk-size", type=int, default=512, help="chunk size used only with --ca-mode optim")
    parser.add_argument("--dummy", action="store_true", help="Use synthetic tensors instead of loading the dataset")
    return parser.parse_args()


def parse_checkpoint_map(entries: list[str]) -> dict[str, Path]:
    checkpoint_map: dict[str, Path] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"invalid checkpoint mapping: {entry}")
        mode, raw_path = entry.split("=", 1)
        checkpoint_map[mode] = Path(raw_path)
    return checkpoint_map


def build_loader(*, split: str, max_samples: int, batch_size: int, seed: int):
    from cr_train.data import DATASET_ID, build_dataloader
    from cr_train.data.dataset import prepare_split
    from cr_train.data.runtime import ensure_split_cache

    cache_root = Path("artifacts") / "gate_eval_cache"
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


def build_dummy_loader(*, max_samples: int, batch_size: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    remaining = max_samples
    while remaining > 0:
        current = min(batch_size, remaining)
        remaining -= current
        yield {
            "sar": torch.rand(current, 2, 16, 16, generator=generator),
            "cloudy": torch.rand(current, 13, 16, 16, generator=generator),
            "target": torch.rand(current, 13, 16, 16, generator=generator),
        }


def save_summary_plot(results: dict[str, dict[str, float]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    modes = list(results.keys())
    metric_specs = [
        ("corr_weighted", "Corr Weighted", False),
        ("clear_mean_g", "Clear Mean G", True),
        ("thick_mean_g", "Thick Mean G", False),
        ("top10_proxy", "Top10 Proxy", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    colors = ["#2563eb", "#f97316", "#16a34a", "#9333ea"]
    for ax, (key, title, lower_is_better) in zip(axes.flat, metric_specs):
        values = [results[mode][key] for mode in modes]
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
    gate: torch.Tensor,
    cloudy: torch.Tensor,
    target: torch.Tensor,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    proxy = compute_band_proxies(cloudy.unsqueeze(0), target.unsqueeze(0))["weighted"][0, 0]
    cloudy_rgb, _, target_rgb = shared_main.normalize_rgb_triplet(cloudy, target, target)
    gate_map = shared_main.normalize_map(prepare_gate_for_eval(gate.unsqueeze(0), target_hw=cloudy.shape[-2:])[0, 0])
    proxy_map = shared_main.normalize_map(proxy)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    panels = (
        ("Cloudy RGB", cloudy_rgb, None),
        ("Target RGB", target_rgb, None),
        ("Weighted Proxy", proxy_map, "magma"),
        ("Gate", gate_map, "viridis"),
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
    loader,
    device: torch.device,
    gate_feat_dim: int,
    gate_prior_weight: float,
    ca_mode: str,
    ca_chunk_size: int,
    num_examples: int,
    output_dir: Path,
    dummy: bool,
) -> dict[str, float]:
    ca = IdentityCA if dummy else {
        "base": ca_base.ConAttn,
        "optim": ca_optim.ConAttn,
        "flash": ca_flash.ConAttn,
    }[ca_mode]
    model_kwargs = {
        "in_channels": 15,
        "out_channels": 13,
        "num_layers": 4 if dummy else 16,
        "feature_sizes": 16 if dummy else 256,
        "gate_mode": mode,
        "gate_feat_dim": gate_feat_dim,
        "gate_prior_weight": gate_prior_weight,
        "ca": ca,
    }
    if not dummy and ca_mode == "optim":
        model_kwargs["ca_kwargs"] = {"chunk_size": ca_chunk_size}
    model = ACA_CRNet(**model_kwargs).to(device)
    model.eval()

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)

    metrics = []
    saved_examples = 0
    with torch.no_grad():
        for batch in loader:
            sar = batch["sar"].to(device)
            cloudy = batch["cloudy"].to(device)
            target = batch["target"].to(device)
            _ = model(sar, cloudy)
            gate = model.last_gate
            if gate is None:
                raise RuntimeError(f"model did not expose last_gate for mode {mode}")
            gate_cpu = gate.detach().cpu()
            cloudy_cpu = cloudy.cpu()
            target_cpu = target.cpu()
            metrics.append(summarize_gate(gate_cpu, cloudy_cpu, target_cpu))

            for index in range(gate_cpu.shape[0]):
                if saved_examples >= num_examples:
                    break
                save_example_figure(
                    mode=mode,
                    sample_index=saved_examples + 1,
                    gate=gate_cpu[index],
                    cloudy=cloudy_cpu[index],
                    target=target_cpu[index],
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
    device = torch.device("cuda" if torch.cuda.is_available() and not args.dummy else "cpu")
    checkpoint_map = parse_checkpoint_map(args.checkpoint)
    output_dir = args.output_dir / args.split / f"samples_{args.max_samples}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, float]] = {}
    for mode in args.gate_modes:
        loader = (
            build_dummy_loader(max_samples=args.max_samples, batch_size=args.batch_size, seed=args.seed)
            if args.dummy
            else build_loader(split=args.split, max_samples=args.max_samples, batch_size=args.batch_size, seed=args.seed)
        )
        results[mode] = evaluate_mode(
            mode=mode,
            checkpoint_path=checkpoint_map.get(mode),
            loader=loader,
            device=device,
            gate_feat_dim=args.gate_feat_dim,
            gate_prior_weight=args.gate_prior_weight,
            ca_mode=args.ca_mode,
            ca_chunk_size=args.ca_chunk_size,
            num_examples=args.num_examples,
            output_dir=output_dir / "examples",
            dummy=args.dummy,
        )

    payload = {
        "gate_modes": list(args.gate_modes),
        "split": args.split,
        "max_samples": args.max_samples,
        "dummy": args.dummy,
        "ca_mode": args.ca_mode,
        "results": results,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    (output_dir / "metrics.json").write_text(text + "\n", encoding="utf-8")
    save_summary_plot(results, output_dir / "summary.png")


if __name__ == "__main__":
    main()
