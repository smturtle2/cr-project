from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

TSNE_GROUPS = ("SAR Feat", "CLD Feat", "CLD Clean", "CLD Cloudy")
TSNE_GROUP_COLORS = {
    "SAR Feat": "#2563eb",
    "CLD Feat": "#7c3aed",
    "CLD Clean": "#06b6d4",
    "CLD Cloudy": "#f97316",
}


def prediction_from_fdt_output(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, torch.Tensor):
        return model_output
    return model_output["prediction"]


def _decomposition_from_fdt_output(
    model_output: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    cld_cloudy = model_output.get(
        "cld_cloudy",
        model_output.get("cloudy_feat", model_output.get("cloud_feat")),
    )
    if cld_cloudy is None:
        raise KeyError("model_output must contain cld_cloudy, cloudy_feat, or cloud_feat")
    cld_clean = model_output.get("cld_clean", model_output.get("clear_feat"))
    if cld_clean is None:
        raise KeyError("model_output must contain cld_clean or clear_feat")
    cld_feat = model_output.get(
        "cld_feat",
        model_output.get("hsi_feat", model_output.get("feature", cld_clean + cld_cloudy)),
    )
    return {
        "prediction": model_output["prediction"],
        "candidate": model_output["candidate"],
        "mask": model_output["mask"],
        "sar_feat": model_output["sar_feat"],
        "cld_feat": cld_feat,
        "cld_clean": cld_clean,
        "cld_cloudy": cld_cloudy,
    }


def _weighted_pca_map(
    feature: torch.Tensor,
    normalize_map: Callable[[torch.Tensor], np.ndarray],
) -> np.ndarray:
    channels, height, width = feature.shape
    tokens = feature.detach().cpu().float().reshape(channels, -1).transpose(0, 1)
    tokens = tokens - tokens.mean(dim=0, keepdim=True)
    if channels == 1:
        scores = tokens[:, 0]
    else:
        _, singular_values, components = torch.linalg.svd(tokens, full_matrices=False)
        variances = singular_values.square()
        if torch.count_nonzero(variances) == 0:
            scores = tokens.new_zeros(tokens.shape[0])
        else:
            signs = torch.sign(components.sum(dim=1))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            scores = tokens @ (components * signs[:, None]).transpose(0, 1)
            weights = torch.sqrt(variances / variances.sum())
            scores = scores @ weights
    return normalize_map(scores.reshape(height, width))


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
    decomposition = _decomposition_from_fdt_output(model_output)
    candidate = decomposition["candidate"]
    mask = decomposition["mask"]
    sar_feat = decomposition["sar_feat"]
    cld_feat = decomposition["cld_feat"]
    cld_clean = decomposition["cld_clean"]
    cld_cloudy = decomposition["cld_cloudy"]
    cloudy_rgb, prediction_rgb, target_rgb = normalize_rgb_triplet(
        cloudy,
        prediction,
        target,
    )
    _, candidate_rgb, _ = normalize_rgb_triplet(cloudy, candidate, target)
    return (
        ("Cloudy RGB", cloudy_rgb, None),
        ("Prediction RGB", prediction_rgb, None),
        ("Target RGB", target_rgb, None),
        ("SAR Mean", normalize_map(sar.mean(dim=0)), "gray"),
        ("Mask Mean", normalize_map(mask.mean(dim=0)), "magma"),
        ("Prediction PCA", _weighted_pca_map(prediction, normalize_map), "magma"),
        ("Target PCA", _weighted_pca_map(target, normalize_map), "magma"),
        ("Candidate RGB", candidate_rgb, None),
        ("SAR Feat PCA", _weighted_pca_map(sar_feat, normalize_map), "magma"),
        ("CLD Feat PCA", _weighted_pca_map(cld_feat, normalize_map), "magma"),
        ("CLD Clean PCA", _weighted_pca_map(cld_clean, normalize_map), "magma"),
        ("CLD Cloudy PCA", _weighted_pca_map(cld_cloudy, normalize_map), "magma"),
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
    sar_feat: torch.Tensor,
    cld_feat: torch.Tensor,
    cld_clear: torch.Tensor,
    cld_cloud: torch.Tensor,
    rng: np.random.Generator,
    points_per_sample: int,
) -> None:
    num_positions = sar_feat.shape[-2] * sar_feat.shape[-1]
    point_count = min(points_per_sample, num_positions)
    indices = rng.choice(num_positions, size=point_count, replace=False)
    grouped_features["SAR Feat"].append(_sample_feature_points(sar_feat, indices))
    grouped_features["CLD Feat"].append(_sample_feature_points(cld_feat, indices))
    grouped_features["CLD Clean"].append(_sample_feature_points(cld_clear, indices))
    grouped_features["CLD Cloudy"].append(_sample_feature_points(cld_cloud, indices))


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
                decomposition = _decomposition_from_fdt_output(model_output)
                sar_feat = decomposition["sar_feat"]
                cld_feat = decomposition["cld_feat"]
                cld_clean = decomposition["cld_clean"]
                cld_cloudy = decomposition["cld_cloudy"]
                batch_size = sar_feat.shape[0]
                for batch_index in range(batch_size):
                    if samples_seen >= sample_count:
                        break
                    _append_tsne_sample_points(
                        grouped_features,
                        sar_feat=sar_feat[batch_index],
                        cld_feat=cld_feat[batch_index],
                        cld_clear=cld_clean[batch_index],
                        cld_cloud=cld_cloudy[batch_index],
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
        labels=("SAR Feat", "CLD Feat"),
        title="Stem Features",
        x_limits=x_limits,
        y_limits=y_limits,
        rng=rng,
        scatter_size=scatter_size,
        scatter_alpha=scatter_alpha,
    )
    _plot_tsne_panel(
        axes[2],
        grouped_embedding,
        labels=("CLD Clean", "CLD Cloudy"),
        title="Cloudy Split",
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
