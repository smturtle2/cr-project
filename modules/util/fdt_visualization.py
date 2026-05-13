from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

TSNE_GROUPS = ("SAR Com", "Cloudy Com", "SAR Comp", "Cloudy Comp")
TSNE_GROUP_COLORS = {
    "SAR Com": "#2563eb",
    "Cloudy Com": "#06b6d4",
    "SAR Comp": "#dc2626",
    "Cloudy Comp": "#f97316",
}


def split_fdt_output(
    model_output: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(model_output, tuple) or len(model_output) != 5:
        raise TypeError(
            "FDT_CRNet must return prediction and four decomposition tensors"
        )
    prediction, sar_com, cld_com, sar_comp, cld_comp = model_output
    return prediction, sar_com, cld_com, sar_comp, cld_comp


def prediction_from_fdt_output(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, torch.Tensor):
        return model_output
    return split_fdt_output(model_output)[0]


def _feature_tokens(feature: torch.Tensor) -> tuple[np.ndarray, int, int]:
    tensor = feature.detach().cpu().float()
    if tensor.ndim != 3:
        raise ValueError("feature must have shape CxHxW")
    channels, height, width = tensor.shape
    tokens = tensor.reshape(channels, height * width).transpose(0, 1).numpy()
    return np.nan_to_num(tokens, copy=False), height, width


def _joint_pc1_maps(
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    first_tokens, height, width = _feature_tokens(first)
    second_tokens, second_height, second_width = _feature_tokens(second)
    if (height, width) != (second_height, second_width):
        raise ValueError("paired features must have the same spatial size")

    merged = np.concatenate([first_tokens, second_tokens], axis=0)
    centered = merged - merged.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    component = components[0]
    if component.sum() < 0:
        component = -component
    projected = centered @ component

    split_at = height * width
    first_map = projected[:split_at].reshape(height, width)
    second_map = projected[split_at:].reshape(height, width)
    return first_map, second_map


def _normalize_paired_maps(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    merged = np.concatenate([first.reshape(-1), second.reshape(-1)], axis=0)
    low, high = np.quantile(merged, [0.02, 0.98])
    normalized = np.clip((merged - low) / max(high - low, 1e-6), 0.0, 1.0)
    split_at = first.size
    return (
        normalized[:split_at].reshape(first.shape),
        normalized[split_at:].reshape(second.shape),
    )


def _pca_heatmaps(
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    return _normalize_paired_maps(*_joint_pc1_maps(first, second))


def _map_ncc(
    first: np.ndarray,
    second: np.ndarray,
    *,
    eps: float = 1e-6,
) -> float:
    first_values = first.reshape(-1) - first.mean()
    second_values = second.reshape(-1) - second.mean()
    numerator = float(np.sum(first_values * second_values))
    denominator = float(
        np.sqrt(np.sum(first_values**2) * np.sum(second_values**2) + eps)
    )
    return numerator / denominator


def _zscore_map(values: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    return (values - values.mean()) / max(float(values.std()), eps)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _pc1_similarity_panel(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    difference = np.abs(_zscore_map(first) - _zscore_map(second))
    return 1.0 - np.clip(difference / 2.0, 0.0, 1.0)


def build_fdt_example_panels(
    *,
    cloudy: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    sar: torch.Tensor,
    model_output: Any | None = None,
    normalize_rgb_triplet: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ],
    normalize_map: Callable[[torch.Tensor], np.ndarray],
):
    _, sar_com, cld_com, sar_comp, cld_comp = split_fdt_output(model_output)
    sar_feat = sar_com + sar_comp
    cld_feat = cld_com + cld_comp
    sar_feat_pca, cld_feat_pca = _pca_heatmaps(sar_feat, cld_feat)
    sar_com_pc1, cld_com_pc1 = _joint_pc1_maps(sar_com, cld_com)
    sar_com_pca, cld_com_pca = _normalize_paired_maps(sar_com_pc1, cld_com_pc1)
    sar_comp_pc1, cld_comp_pc1 = _joint_pc1_maps(sar_comp, cld_comp)
    sar_comp_pca, cld_comp_pca = _normalize_paired_maps(sar_comp_pc1, cld_comp_pc1)
    common_ncc = _map_ncc(sar_com_pc1, cld_com_pc1)
    common_match = _pc1_similarity_panel(sar_com_pc1, cld_com_pc1) * _clip01(common_ncc)
    comp_ncc_squared = _map_ncc(sar_comp_pc1, cld_comp_pc1) ** 2
    comp_leak = _pc1_similarity_panel(sar_comp_pc1, cld_comp_pc1) * _clip01(comp_ncc_squared)
    cloudy_rgb, prediction_rgb, target_rgb = normalize_rgb_triplet(
        cloudy,
        prediction,
        target,
    )
    return (
        ("Cloudy RGB", cloudy_rgb, None),
        ("Prediction RGB", prediction_rgb, None),
        ("Target RGB", target_rgb, None),
        ("SAR Mean", normalize_map(sar.mean(dim=0)), "gray"),
        (f"Com Match  {common_ncc:+.2f}  ↑ Good", common_match, "magma", 0.0, 1.0),
        ("SAR Feat PCA", sar_feat_pca, "viridis"),
        ("SAR Com PCA", sar_com_pca, "viridis"),
        ("SAR Comp PCA", sar_comp_pca, "viridis"),
        (f"Comp Leak  {comp_ncc_squared: .2f}  ↓ Good", comp_leak, "magma", 0.0, 1.0),
        ("Cloudy Feat PCA", cld_feat_pca, "viridis"),
        ("Cloudy Com PCA", cld_com_pca, "viridis"),
        ("Cloudy Comp PCA", cld_comp_pca, "viridis"),
    )


def _sample_feature_points(
    feature: torch.Tensor,
    indices: np.ndarray,
) -> np.ndarray:
    channels, height, width = feature.shape
    del height, width
    tokens = feature.detach().cpu().float().reshape(channels, -1).transpose(0, 1)
    return np.nan_to_num(tokens[indices].numpy(), copy=False)


def _append_tsne_sample_points(
    grouped_features: dict[str, list[np.ndarray]],
    *,
    sar_com: torch.Tensor,
    cld_com: torch.Tensor,
    sar_comp: torch.Tensor,
    cld_comp: torch.Tensor,
    rng: np.random.Generator,
    points_per_sample: int,
) -> None:
    num_positions = sar_com.shape[-2] * sar_com.shape[-1]
    point_count = min(points_per_sample, num_positions)
    indices = rng.choice(num_positions, size=point_count, replace=False)
    grouped_features["SAR Com"].append(_sample_feature_points(sar_com, indices))
    grouped_features["Cloudy Com"].append(_sample_feature_points(cld_com, indices))
    grouped_features["SAR Comp"].append(_sample_feature_points(sar_comp, indices))
    grouped_features["Cloudy Comp"].append(_sample_feature_points(cld_comp, indices))


def _cap_group_points(
    features: np.ndarray,
    *,
    rng: np.random.Generator,
    max_points_per_group: int,
) -> np.ndarray:
    if features.shape[0] <= max_points_per_group:
        return features
    indices = rng.choice(
        features.shape[0],
        size=max_points_per_group,
        replace=False,
    )
    return features[indices]


def _collect_tsne_features(
    dataloader: Any,
    predict_fn: Callable[[Mapping[str, Any]], Any],
    *,
    progress_label: str,
    sample_count: int,
    max_points_per_group: int,
    random_seed: int,
    show_progress: bool,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    grouped_features: dict[str, list[np.ndarray]] = {name: [] for name in TSNE_GROUPS}
    points_per_sample = max(
        1,
        (max_points_per_group + sample_count - 1) // sample_count,
    )

    samples_seen = 0
    iterator = iter(dataloader)
    progress = None
    try:
        if show_progress:
            try:
                from tqdm.auto import tqdm

                progress = tqdm(
                    total=sample_count,
                    desc=progress_label,
                    unit="sample",
                    leave=False,
                )
            except ImportError:
                progress = None
        with torch.no_grad():
            while samples_seen < sample_count:
                try:
                    batch = next(iterator)
                except StopIteration:
                    break

                model_output = predict_fn(batch)
                _, sar_com, cld_com, sar_comp, cld_comp = split_fdt_output(
                    model_output
                )
                batch_size = sar_com.shape[0]
                for batch_index in range(batch_size):
                    if samples_seen >= sample_count:
                        break
                    _append_tsne_sample_points(
                        grouped_features,
                        sar_com=sar_com[batch_index],
                        cld_com=cld_com[batch_index],
                        sar_comp=sar_comp[batch_index],
                        cld_comp=cld_comp[batch_index],
                        rng=rng,
                        points_per_sample=points_per_sample,
                    )
                    samples_seen += 1
                    if progress is not None:
                        progress.update(1)
    finally:
        if progress is not None:
            progress.close()
        del iterator

    return {
        name: _cap_group_points(
            np.concatenate(chunks, axis=0),
            rng=rng,
            max_points_per_group=max_points_per_group,
        )
        for name, chunks in grouped_features.items()
        if chunks
    }


def _standardize_features(features: np.ndarray) -> np.ndarray:
    features = features.astype(np.float32, copy=False)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / np.maximum(std, 1e-6)


def _fit_tsne_embedding(
    grouped_features: dict[str, np.ndarray],
    *,
    pre_pca_dim: int,
    random_seed: int,
) -> dict[str, np.ndarray]:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    labels = []
    arrays = []
    for name in TSNE_GROUPS:
        features = grouped_features.get(name)
        if features is None or features.shape[0] == 0:
            continue
        labels.extend([name] * features.shape[0])
        arrays.append(features)
    if not arrays:
        return {}

    features = _standardize_features(np.concatenate(arrays, axis=0))
    pca_dim = min(pre_pca_dim, features.shape[0] - 1, features.shape[1])
    if pca_dim >= 2:
        features = PCA(n_components=pca_dim, random_state=random_seed).fit_transform(
            features
        )

    perplexity = min(30.0, max(5.0, (features.shape[0] - 1) / 3.0))
    embedding = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=random_seed,
    ).fit_transform(features)

    grouped_embedding: dict[str, list[np.ndarray]] = {name: [] for name in TSNE_GROUPS}
    for label, point in zip(labels, embedding):
        grouped_embedding[label].append(point)
    return {
        name: np.stack(points, axis=0)
        for name, points in grouped_embedding.items()
        if points
    }


def _embedding_limits(grouped_embedding: dict[str, np.ndarray]):
    merged = np.concatenate(list(grouped_embedding.values()), axis=0)
    x_min, y_min = merged.min(axis=0)
    x_max, y_max = merged.max(axis=0)
    x_margin = max((x_max - x_min) * 0.06, 1e-3)
    y_margin = max((y_max - y_min) * 0.06, 1e-3)
    return (
        (x_min - x_margin, x_max + x_margin),
        (y_min - y_margin, y_max + y_margin),
    )


def _plot_tsne_panel(
    ax,
    grouped_embedding: dict[str, np.ndarray],
    *,
    labels: tuple[str, ...],
    title: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    rng: np.random.Generator,
    scatter_size: float,
    scatter_alpha: float,
) -> None:
    panel_points = []
    panel_colors = []
    for label in labels:
        points = grouped_embedding.get(label)
        if points is None:
            continue
        color = TSNE_GROUP_COLORS[label]
        panel_points.append(points)
        panel_colors.extend([color] * points.shape[0])

    if panel_points:
        points = np.concatenate(panel_points, axis=0)
        colors = np.asarray(panel_colors)
        order = rng.permutation(points.shape[0])
        ax.scatter(
            points[order, 0],
            points[order, 1],
            s=scatter_size,
            alpha=scatter_alpha,
            linewidths=0,
            edgecolors="none",
            c=colors[order],
            rasterized=True,
        )

    ax.set_title(title)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")


def _save_tsne_scatter_figure(
    grouped_features: dict[str, np.ndarray],
    *,
    path: Path,
    title: str,
    pre_pca_dim: int,
    random_seed: int,
    scatter_size: float,
    scatter_alpha: float,
) -> Path | None:
    import matplotlib.pyplot as plt

    print("fitting t-SNE scatter...")
    grouped_embedding = _fit_tsne_embedding(
        grouped_features,
        pre_pca_dim=pre_pca_dim,
        random_seed=random_seed,
    )
    if len(grouped_embedding) < 2:
        return None

    import matplotlib.patches as mpatches

    x_limits, y_limits = _embedding_limits(grouped_embedding)
    rng = np.random.default_rng(random_seed)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=11)
    _plot_tsne_panel(
        axes[0],
        grouped_embedding,
        labels=TSNE_GROUPS,
        title="All",
        x_limits=x_limits,
        y_limits=y_limits,
        rng=rng,
        scatter_size=scatter_size,
        scatter_alpha=scatter_alpha,
    )
    _plot_tsne_panel(
        axes[1],
        grouped_embedding,
        labels=("SAR Com", "Cloudy Com"),
        title="Com Only",
        x_limits=x_limits,
        y_limits=y_limits,
        rng=rng,
        scatter_size=scatter_size,
        scatter_alpha=scatter_alpha,
    )
    _plot_tsne_panel(
        axes[2],
        grouped_embedding,
        labels=("SAR Comp", "Cloudy Comp"),
        title="Comp Only",
        x_limits=x_limits,
        y_limits=y_limits,
        rng=rng,
        scatter_size=scatter_size,
        scatter_alpha=scatter_alpha,
    )
    handles = [
        mpatches.Patch(color=TSNE_GROUP_COLORS[label], label=label)
        for label in TSNE_GROUPS
        if label in grouped_embedding
    ]
    if handles:
        fig.legend(
            handles=handles,
            frameon=False,
            fontsize=8,
            loc="center right",
            bbox_to_anchor=(0.995, 0.5),
        )
    fig.tight_layout()
    fig.subplots_adjust(top=0.86, right=0.86)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved t-SNE scatter: {path}")
    return path


def save_fdt_tsne_scatter(
    *,
    dataloader: Any,
    predict_fn: Callable[[Mapping[str, Any]], Any],
    path: Path,
    title: str,
    is_primary: Callable[[], bool],
    sample_count: int = 128,
    max_points_per_group: int = 4096,
    pre_pca_dim: int = 50,
    random_seed: int = 42,
    scatter_size: float = 3,
    scatter_alpha: float = 0.12,
    show_progress: bool = True,
) -> Path | None:
    if not is_primary():
        return None
    grouped_features = _collect_tsne_features(
        dataloader,
        predict_fn,
        progress_label="t-SNE samples",
        sample_count=sample_count,
        max_points_per_group=max_points_per_group,
        random_seed=random_seed,
        show_progress=show_progress,
    )
    return _save_tsne_scatter_figure(
        grouped_features,
        path=path,
        title=title,
        pre_pca_dim=pre_pca_dim,
        random_seed=random_seed,
        scatter_size=scatter_size,
        scatter_alpha=scatter_alpha,
    )


def load_fdt_checkpoint_model(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
        raise TypeError("checkpoint must contain a model state dict")
    state_owner = model.module if hasattr(model, "module") else model
    state_owner.load_state_dict(checkpoint["model"])
    return int(checkpoint.get("epoch", 0))


def save_fdt_tsne_for_splits(
    *,
    dataloaders_by_split: Mapping[str, Any],
    predict_fn: Callable[[Mapping[str, Any]], Any],
    output_dir: Path,
    is_primary: Callable[[], bool],
    sample_count: int = 128,
    max_points_per_group: int = 4096,
    pre_pca_dim: int = 50,
    random_seed: int = 42,
    scatter_size: float = 3,
    scatter_alpha: float = 0.12,
    show_progress: bool = True,
) -> None:
    if not dataloaders_by_split:
        return
    for split, dataloader in dataloaders_by_split.items():
        save_fdt_tsne_scatter(
            dataloader=dataloader,
            predict_fn=predict_fn,
            path=output_dir / split / f"{split}_tsne_scatter.png",
            title=f"{split} t-SNE scatter",
            is_primary=is_primary,
            sample_count=sample_count,
            max_points_per_group=max_points_per_group,
            pre_pca_dim=pre_pca_dim,
            random_seed=random_seed,
            scatter_size=scatter_size,
            scatter_alpha=scatter_alpha,
            show_progress=show_progress,
        )
