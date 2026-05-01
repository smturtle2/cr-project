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
from modules.model.cafm.ACA_CRNet import ACA_CRNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Density estimator offline evaluation")
    parser.add_argument(
        "--density-modes",
        nargs="+",
        choices=("cosine", "cosine_prior"),
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
        if lower_is_better:
            best_value = min(values)
        else:
            best_value = max(values)
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
    model = ACA_CRNet(use_cafm=True, use_sdi=False, density_mode=mode).to(device)
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
