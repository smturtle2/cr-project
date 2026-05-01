from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main as shared_main
from modules.metrics.gate_eval import compute_band_proxies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SEN12-CR proxy cloud influence maps")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low-percentile", type=float, default=2.0)
    parser.add_argument("--high-percentile", type=float, default=98.0)
    parser.add_argument("--overlay-alpha", type=float, default=0.35)
    parser.add_argument("--contour-threshold", type=float, default=0.8)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "proxy_map_inspection")
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts") / "proxy_map_cache")
    parser.add_argument("--dummy", action="store_true", help="Use synthetic tensors instead of SEN12-CR")
    return parser.parse_args()


def normalize_proxy_map(
    proxy: torch.Tensor,
    *,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> tuple[np.ndarray, dict[str, float]]:
    if proxy.ndim != 2:
        raise ValueError(f"proxy must be a 2D tensor, got shape {tuple(proxy.shape)}")
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")

    array = proxy.detach().cpu().float().numpy()
    low, high = np.percentile(array, [low_percentile, high_percentile])
    scale = max(float(high - low), 1e-6)
    normalized = np.clip((array - low) / scale, 0.0, 1.0)
    stats = {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "low_value": float(low),
        "high_value": float(high),
    }
    return normalized, stats


def colorize_proxy_map(proxy_norm: np.ndarray, *, cmap_name: str = "magma") -> np.ndarray:
    import matplotlib.pyplot as plt

    colormap = plt.get_cmap(cmap_name)
    return colormap(np.clip(proxy_norm, 0.0, 1.0))[..., :3]


def blend_overlay(base_rgb: np.ndarray, heatmap_rgb: np.ndarray, *, alpha: float) -> np.ndarray:
    if base_rgb.shape != heatmap_rgb.shape:
        raise ValueError(f"base and heatmap shapes must match, got {base_rgb.shape} and {heatmap_rgb.shape}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return np.clip((1.0 - alpha) * base_rgb + alpha * heatmap_rgb, 0.0, 1.0)


def build_loader(*, split: str, max_samples: int, batch_size: int, seed: int, cache_dir: Path):
    from cr_train.data import DATASET_ID, build_dataloader
    from cr_train.data.dataset import prepare_split
    from cr_train.data.runtime import ensure_split_cache

    ensure_split_cache(
        split=split,
        dataset_name=DATASET_ID,
        revision=None,
        max_samples=max_samples,
        seed=seed,
        cache_root=cache_dir,
    )
    prepared = prepare_split(
        split=split,
        dataset_name=DATASET_ID,
        revision=None,
        max_samples=max_samples,
        seed=seed,
        epoch=0,
        training=False,
        cache_root=cache_dir,
    )
    return build_dataloader(
        prepared,
        batch_size=batch_size,
        num_workers=0,
        training=False,
        seed=seed,
        epoch=0,
        include_metadata=True,
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
        target = torch.rand(current, 13, 64, 64, generator=generator)
        cloudy = target.clone()
        cloudy[:, 1, 12:36, 16:44] += 0.35
        cloudy[:, 3, 12:36, 16:44] += 0.20
        cloudy[:, 10, 18:48, 22:54] += 0.45
        cloudy[:, 11, 18:48, 22:54] += 0.25
        cloudy = cloudy.clamp(0.0, 1.0)
        yield {
            "sar": torch.rand(current, 2, 64, 64, generator=generator),
            "cloudy": cloudy,
            "target": target,
            "metadata": {"dummy": [True for _ in range(current)]},
        }


def _metadata_value(metadata: Any, key: str, index: int) -> str:
    if not isinstance(metadata, dict):
        return ""
    values = metadata.get(key, [])
    if not isinstance(values, (list, tuple)) or index >= len(values):
        return ""
    return str(values[index])


def _sample_title(*, split: str, sample_number: int, metadata: Any, batch_index: int) -> str:
    season = _metadata_value(metadata, "season", batch_index)
    scene = _metadata_value(metadata, "scene", batch_index)
    patch = _metadata_value(metadata, "patch", batch_index)
    if season or scene or patch:
        return f"{split} sample {sample_number} | {season}/scene_{scene}/patch_{patch}"
    return f"{split} sample {sample_number}"


def save_proxy_figure(
    *,
    cloudy: torch.Tensor,
    target: torch.Tensor,
    output_path: Path,
    title: str,
    low_percentile: float,
    high_percentile: float,
    overlay_alpha: float,
    contour_threshold: float,
) -> dict[str, float]:
    import matplotlib.pyplot as plt

    proxy = compute_band_proxies(cloudy.unsqueeze(0), target.unsqueeze(0))["weighted"][0, 0]
    proxy_norm, stats = normalize_proxy_map(
        proxy,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
    )
    heatmap_rgb = colorize_proxy_map(proxy_norm)
    cloudy_rgb, _, target_rgb = shared_main.normalize_rgb_triplet(cloudy, target, target)
    overlay = blend_overlay(cloudy_rgb, heatmap_rgb, alpha=overlay_alpha)
    contour_mask = proxy_norm >= contour_threshold

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    panels = (
        ("Cloudy RGB", cloudy_rgb, None),
        ("GT RGB", target_rgb, None),
        ("Proxy Cloud Influence", proxy_norm, "magma"),
        ("Cloudy + Proxy Overlay", overlay, None),
    )
    for ax, (panel_title, image, cmap) in zip(axes, panels):
        if cmap is None:
            ax.imshow(image)
        else:
            im = ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Relative influence", fontsize=8)
        if panel_title == "Cloudy + Proxy Overlay" and np.any(contour_mask):
            ax.contour(contour_mask, levels=[0.5], colors="cyan", linewidths=0.8)
        ax.set_title(panel_title)
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return stats


def main() -> None:
    args = parse_args()
    shared_main.seed_everything(args.seed)
    output_dir = args.output_dir / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = (
        build_dummy_loader(max_samples=args.max_samples, batch_size=args.batch_size, seed=args.seed)
        if args.dummy
        else build_loader(
            split=args.split,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            seed=args.seed,
            cache_dir=args.cache_dir,
        )
    )

    samples = []
    saved = 0
    for batch in loader:
        cloudy_batch = batch["cloudy"]
        target_batch = batch["target"]
        metadata = batch.get("metadata", {})
        for batch_index in range(cloudy_batch.shape[0]):
            if saved >= args.max_samples:
                break
            saved += 1
            output_path = output_dir / f"sample_{saved:03d}.png"
            title = _sample_title(
                split=args.split,
                sample_number=saved,
                metadata=metadata,
                batch_index=batch_index,
            )
            stats = save_proxy_figure(
                cloudy=cloudy_batch[batch_index],
                target=target_batch[batch_index],
                output_path=output_path,
                title=title,
                low_percentile=args.low_percentile,
                high_percentile=args.high_percentile,
                overlay_alpha=args.overlay_alpha,
                contour_threshold=args.contour_threshold,
            )
            samples.append({"index": saved, "file": str(output_path), "title": title, "proxy": stats})
        if saved >= args.max_samples:
            break

    payload = {
        "split": args.split,
        "max_samples": args.max_samples,
        "saved_samples": saved,
        "seed": args.seed,
        "dummy": args.dummy,
        "low_percentile": args.low_percentile,
        "high_percentile": args.high_percentile,
        "overlay_alpha": args.overlay_alpha,
        "contour_threshold": args.contour_threshold,
        "samples": samples,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved {saved} proxy inspection figures to {output_dir}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
