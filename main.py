from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from huggingface_hub import get_token
from torch import nn
from torch.optim.lr_scheduler import LRScheduler

from cr_train import Trainer
from cr_train.data import DATASET_ID, build_dataloader
from cr_train.data.dataset import prepare_split
from cr_train.data.runtime import ensure_split_cache


# Sentinel-2 13채널 중 사람이 보기 쉬운 RGB 조합(B4, B3, B2) 인덱스다.
RGB_CHANNELS = (3, 2, 1)
EXAMPLE_SPLITS = ("train", "validation", "test")
STAGE_PLOT_STYLES = {
    "train": {
        "label": "Train",
        "color": "#1f77b4",
        "linestyle": "-",
        "marker": "o",
        "markersize": 4.5,
        "linewidth": 2.2,
    },
    "val": {
        "label": "Validation",
        "color": "#ff7f0e",
        "linestyle": "-",
        "marker": "s",
        "markersize": 4.5,
        "linewidth": 2.2,
    },
    "test": {
        "label": "Test",
        "color": "#2ca02c",
        "linestyle": "None",
        "marker": "D",
        "markersize": 6.5,
        "linewidth": 0.0,
    },
}

# Trainer가 기대하는 `(prediction, batch)` 계약을 타입으로도 분명히 적어 둔다.
Batch = dict[str, Any]
LossFn = Callable[[torch.Tensor, Batch], torch.Tensor]
MetricFn = Callable[[torch.Tensor, Batch], torch.Tensor]
StepRecord = Mapping[str, Any]
SchedulerTiming = Literal["after_validation", "before_optimizer_step", "after_optimizer_step"]


@dataclass(slots=True, frozen=True)
class BestEpochSelector:
    name: str
    mode: Literal["min", "max"]
    score_fn: Callable[[StepRecord], float]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("BestEpochSelector.name must not be empty")
        if self.mode not in {"min", "max"}:
            raise ValueError("BestEpochSelector.mode must be either 'min' or 'max'")
        if not callable(self.score_fn):
            raise TypeError("BestEpochSelector.score_fn must be callable")


@dataclass(slots=True)
class BestState:
    epoch: int
    score: float


@dataclass(slots=True, frozen=True)
class ExampleSaveConfig:
    splits: tuple[str, ...]
    max_samples_by_split: Mapping[str, int | None]
    num_examples: int


def seed_everything(seed: int) -> None:
    # 모델 가중치를 만드는 시점부터 재현 가능하도록 Trainer 생성 전에 한 번 더 고정한다.
    # Trainer 내부도 epoch마다 다시 시드를 관리하지만, 모델 초기화는 main 쪽 책임이다.
    # 최신 cr-train은 epoch seed는 유지하되 deterministic 알고리즘 강제는 하지 않으므로
    # 여기서도 RNG 시드만 맞추고 CUDA 비결정 연산까지 막지는 않는다.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_hf_auth_status() -> None:
    # 데이터셋은 Hugging Face에서 오므로 토큰이 잡혀 있는지만 가볍게 알려 준다.
    # 인증 자체를 여기서 처리하지는 않고, 상태만 보여 준다.
    print(f"HF auth: {'configured' if get_token() else 'missing'}")


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model() -> nn.Module:
    # 사용자 프로젝트에서 실제 모델을 연결할 지점이다.
    # cr-train은 `forward(sar, cloudy)` 시그니처를 기대하므로 그 규약만 맞추면 된다.
    raise NotImplementedError("build_model()을 구현하세요.")


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    # 사용자 프로젝트에서 실제 optimizer를 연결할 지점이다.
    raise NotImplementedError("build_optimizer()를 구현하세요.")


def build_loss() -> LossFn:
    # loss도 모델/optimizer와 마찬가지로 프로젝트 쪽 교체 포인트다.
    # Trainer가 기대하는 `(prediction, batch)` 계약을 여기서 맞춰 준다.
    l1_loss = nn.L1Loss()

    def loss_fn(prediction: torch.Tensor, batch: Batch) -> torch.Tensor:
        return l1_loss(prediction, batch["target"])

    return loss_fn


def build_metrics() -> dict[str, MetricFn]:
    # metric 역시 Trainer에 직접 넘기는 사용자 정의 지점이다.
    # 기본 예시는 target 기준 MAE 하나만 사용한다.
    def mae(prediction: torch.Tensor, batch: Batch) -> torch.Tensor:
        return torch.mean(torch.abs(prediction - batch["target"]))

    return {"mae": mae}


def build_best_epoch_selector() -> BestEpochSelector:
    # 기본 best 기준은 validation loss 최소값이다.
    return BestEpochSelector(
        name="val_loss",
        mode="min",
        score_fn=lambda record: float(record["val"]["loss"]),
    )


def build_scheduler(optimizer: torch.optim.Optimizer) -> LRScheduler | None:
    # scheduler가 필요하면 optimizer를 받아 여기서 연결한다.
    # 기본값은 scheduler 없이 학습한다.
    del optimizer
    return None


def build_scheduler_monitor() -> str | None:
    # ReduceLROnPlateau를 쓸 때만 monitor 경로를 넘기면 된다.
    # 최신 cr-train에서는 scheduler_timing="after_validation"일 때만 유효하다.
    # 기본값 None이면 cr-train 기본 monitor(`val.loss`)를 따른다.
    return None


def build_trainer(
    *,
    batch_size: int = 8,
    accum_steps: int = 1,
    seed: int = 42,
    max_epochs: int = 10,
    train_max_samples: int | None = 16384,
    val_max_samples: int | None = 2048,
    test_max_samples: int | None = 2048,
    streaming: bool = True,
    output_dir: str | Path = Path("artifacts"),
    cache_dir: str | Path | None = None,
    num_workers: int | str = "auto",
    multiprocessing_context: str | None = None,
    scheduler_timing: SchedulerTiming = "after_validation",
    train_crop_size: int | None = 128,
    train_random_flip: bool = True,
    train_random_rot90: bool = True,
) -> Trainer:
    # main.py와 그 소비자들이 같은 Trainer 구성을 공유하도록 공용 생성 helper로 둔다.
    output_dir = Path(output_dir)
    seed_everything(seed)
    print_hf_auth_status()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model().to(resolve_device())
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)
    return Trainer(
        model=model,
        optimizer=optimizer,
        loss=build_loss(),
        metrics=build_metrics(),
        scheduler=scheduler,
        scheduler_timing=scheduler_timing,
        scheduler_monitor=build_scheduler_monitor(),
        max_train_samples=train_max_samples,
        max_val_samples=val_max_samples,
        max_test_samples=test_max_samples,
        output_dir=output_dir,
        cache_dir=cache_dir,
        batch_size=batch_size,
        accum_steps=accum_steps,
        epochs=max_epochs,
        seed=seed,
        streaming=streaming,
        num_workers=num_workers,
        multiprocessing_context=multiprocessing_context,
        train_crop_size=train_crop_size,
        train_random_flip=train_random_flip,
        train_random_rot90=train_random_rot90,
    )


def _normalize_learning_rates(values: Any) -> list[float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []

    learning_rates: list[float] = []
    for value in values:
        learning_rates.append(float(value))
    return learning_rates


def _write_learning_rates(
    row: dict[str, int | float],
    *,
    stage: Literal["train", "val", "test"],
    values: Any,
) -> None:
    learning_rates = _normalize_learning_rates(values)
    if not learning_rates:
        return

    if len(learning_rates) == 1:
        row[f"{stage}_lr"] = learning_rates[0]
        return

    for index, learning_rate in enumerate(learning_rates):
        row[f"{stage}_lr_group_{index}"] = learning_rate


def flatten_record(record: Mapping[str, Any], *, global_step: int) -> dict[str, int | float]:
    # epoch 단위 결과를 plot용 row 하나로 바꾼다.
    # Trainer가 이미 사람이 읽기 좋은 요약을 출력하므로, 여기서는 시각화에 필요한 값만 남긴다.
    # 최신 cr-train에서 global_step은 micro-batch가 아니라 optimizer update 수다.
    # 최신 cr-train은 epoch 전체 소요 시간도 함께 주므로, 표/그래프에서 같이 볼 수 있게 보존한다.
    row: dict[str, int | float] = {
        "epoch": int(record["epoch"]),
        "global_step": int(global_step),
    }

    if "elapsed_sec" in record:
        row["elapsed_sec"] = float(record["elapsed_sec"])

    for stage in ("train", "val", "test"):
        summary = record.get(stage)
        if not summary:
            continue

        write_stage_summary(row, stage=stage, summary=summary)

    return row


def append_history(
    history: list[dict[str, int | float]],
    record: Mapping[str, Any],
    *,
    global_step: int,
) -> None:
    history.append(flatten_record(record, global_step=global_step))


def write_stage_summary(
    row: dict[str, int | float],
    *,
    stage: Literal["train", "val", "test"],
    summary: Mapping[str, Any],
) -> None:
    if "loss" in summary:
        row[f"{stage}_loss"] = float(summary["loss"])

    if "lr" in summary:
        _write_learning_rates(row, stage=stage, values=summary["lr"])

    metrics = summary.get("metrics", {})
    if isinstance(metrics, Mapping):
        for name, value in metrics.items():
            row[f"{stage}_{name}"] = float(value)


def load_history_from_metrics_jsonl(path: str | Path) -> list[dict[str, int | float]]:
    # train/validation/test 이벤트 로그를 plot용 sparse epoch row로 다시 조립한다.
    # metrics.jsonl epoch 요약에는 optimizer update 기준 global_step이 따로 없으므로
    # 여기서는 실제 step 복원이 아니라 epoch 번호를 fallback 값으로 넣는다.
    path = Path(path)
    merged_rows: dict[int, dict[str, int | float]] = {}
    test_rows: list[dict[str, int | float]] = []
    stage_by_kind: dict[str, Literal["train", "val", "test"]] = {
        "train_epoch": "train",
        "validation": "val",
        "test": "test",
    }

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            stage = stage_by_kind.get(str(record.get("kind")))
            if stage is None:
                continue

            epoch = int(record["epoch"])
            row = {
                "epoch": epoch,
                "global_step": epoch,
            }
            write_stage_summary(row, stage=stage, summary=record)

            if stage == "test":
                test_rows.append(row)
                continue

            merged = merged_rows.setdefault(
                epoch,
                {
                    "epoch": epoch,
                    "global_step": epoch,
                },
            )
            merged.update(row)

    history = [merged_rows[epoch] for epoch in sorted(merged_rows)]
    history.extend(sorted(test_rows, key=lambda row: int(row["epoch"])))
    return history


def split_history_metric_key(key: str) -> tuple[Literal["train", "val", "test"], str] | None:
    for stage in ("train", "val", "test"):
        prefix = f"{stage}_"
        if key.startswith(prefix) and len(key) > len(prefix):
            return stage, key[len(prefix) :]
    return None


def format_metric_label(metric: str) -> str:
    if metric == "lr":
        return "LR"
    if metric.startswith("lr_group_"):
        group_index = metric.removeprefix("lr_group_")
        if group_index.isdigit():
            return f"LR (Group {int(group_index)})"

    labels = {
        "loss": "Loss",
        "mae": "MAE",
        "psnr": "PSNR",
        "sam": "SAM",
        "ssim": "SSIM",
    }
    return labels.get(metric, metric.replace("_", " ").title())


def build_metric_plot_path(path: Path, metric: str) -> Path:
    sanitized_metric = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in metric)
    return path.with_name(f"{path.stem}_{sanitized_metric}.png")


def score_epoch(record: StepRecord, *, selector: BestEpochSelector) -> float:
    score = float(selector.score_fn(record))
    if not math.isfinite(score):
        raise ValueError(f"best epoch selector '{selector.name}' returned a non-finite score")
    return score


def is_better_score(
    score: float,
    best_score: float | None,
    *,
    mode: Literal["min", "max"],
) -> bool:
    if best_score is None:
        return True
    if mode == "min":
        return score < best_score
    if mode == "max":
        return score > best_score
    raise ValueError(f"unsupported best score mode: {mode}")


def best_metadata_path(output_dir: Path) -> Path:
    return output_dir / "best.json"


def best_checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "best.pt"


def _selector_matches_best_payload(payload: Mapping[str, Any], *, selector: BestEpochSelector) -> bool:
    return (
        str(payload["selector_name"]) == selector.name
        and str(payload["selector_mode"]) == selector.mode
    )


def _build_best_payload(
    *,
    epoch: int,
    score: float,
    selector: BestEpochSelector,
    checkpoint_path: Path,
) -> dict[str, str | int | float]:
    checkpoint_text = str(checkpoint_path)
    return {
        "epoch": int(epoch),
        "score": float(score),
        "checkpoint_path": checkpoint_text,
        "selector_name": selector.name,
        "selector_mode": selector.mode,
        "source_checkpoint_path": checkpoint_text,
    }


def load_best_state(output_dir: Path, *, selector: BestEpochSelector) -> BestState | None:
    metadata_path = best_metadata_path(output_dir)
    if not metadata_path.exists():
        return None

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not _selector_matches_best_payload(payload, selector=selector):
        print(
            "ignored existing best metadata:",
            f"path={metadata_path}",
            f"selector={payload['selector_name']}/{payload['selector_mode']}",
            f"expected={selector.name}/{selector.mode}",
        )
        return None

    return BestState(
        epoch=int(payload["epoch"]),
        score=float(payload["score"]),
    )


def save_best_state(
    output_dir: Path,
    *,
    epoch: int,
    score: float,
    selector: BestEpochSelector,
    checkpoint_path: Path,
) -> BestState:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _build_best_payload(
        epoch=epoch,
        score=score,
        selector=selector,
        checkpoint_path=checkpoint_path,
    )
    best_metadata_path(output_dir).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "updated best checkpoint:",
        f"epoch={epoch}",
        f"{selector.name}={score:.6f}",
        f"path={checkpoint_path}",
    )
    return BestState(
        epoch=int(epoch),
        score=float(score),
    )


def _collect_history_metric_names(history: Sequence[Mapping[str, int | float]]) -> list[str]:
    grouped_metrics: dict[str, set[str]] = {}
    for row in history:
        for key in row:
            if key in {"epoch", "global_step", "elapsed_sec"}:
                continue
            match = split_history_metric_key(key)
            if match is None:
                continue
            stage, metric = match
            grouped_metrics.setdefault(metric, set()).add(stage)

    def metric_sort_key(metric: str) -> tuple[int, int, str]:
        if metric == "loss":
            return (0, 0, metric)
        if metric == "lr":
            return (1, 0, metric)
        if metric.startswith("lr_group_"):
            group_index = metric.removeprefix("lr_group_")
            if group_index.isdigit():
                return (1, int(group_index) + 1, metric)
            return (1, 10_000, metric)
        return (2, 0, metric)

    return sorted(grouped_metrics, key=metric_sort_key)


def _history_metric_values(
    history: Sequence[Mapping[str, int | float]],
    *,
    stage: Literal["train", "val", "test"],
    metric: str,
) -> list[tuple[int, float]]:
    return [
        (int(row["epoch"]), float(row[f"{stage}_{metric}"]))
        for row in history
        if f"{stage}_{metric}" in row
    ]


def _plot_metric_series(
    ax,
    history: Sequence[Mapping[str, int | float]],
    *,
    metric: str,
) -> bool:
    has_series = False
    for stage in ("train", "val", "test"):
        values = _history_metric_values(history, stage=stage, metric=metric)
        if not values:
            continue
        has_series = True
        epochs, series = zip(*values, strict=False)
        ax.plot(
            epochs,
            series,
            color=STAGE_PLOT_STYLES[stage]["color"],
            label=STAGE_PLOT_STYLES[stage]["label"],
            linestyle=STAGE_PLOT_STYLES[stage]["linestyle"],
            marker=STAGE_PLOT_STYLES[stage]["marker"],
            markersize=STAGE_PLOT_STYLES[stage]["markersize"],
            linewidth=STAGE_PLOT_STYLES[stage]["linewidth"],
        )
    return has_series


def _style_metric_axis(ax, *, metric: str, has_series: bool) -> None:
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("value")
    ax.set_title(format_metric_label(metric))
    ax.set_xlabel("epoch")
    if has_series:
        ax.legend(frameon=False)


def _save_metric_plot(history: Sequence[Mapping[str, int | float]], *, path: Path, metric: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.2))
    has_series = _plot_metric_series(ax, history, metric=metric)
    _style_metric_axis(ax, metric=metric, has_series=has_series)

    output_path = build_metric_plot_path(path, metric)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved plot: {output_path}")


def save_history_plot(history: list[dict[str, int | float]], path: Path) -> None:
    # metric마다 독립 PNG를 남겨 두면 학습이 길어져도 필요한 지표만 바로 열어 보기 쉽다.
    if not history:
        return

    metric_names = _collect_history_metric_names(history)
    if not metric_names:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    for metric in metric_names:
        _save_metric_plot(history, path=path, metric=metric)


def build_loader(
    trainer: Trainer,
    *,
    split: str,
    max_samples: int | None,
    training: bool,
    epoch_index: int,
):
    # 예시 이미지를 만들거나 notebook에서 배치 shape를 확인할 때만 별도 loader가 필요하다.
    # 이때도 데이터 준비 자체는 cr-train의 공개 data API를 그대로 사용한다.
    if trainer.streaming:
        ensure_split_cache(
            split=split,
            dataset_name=DATASET_ID,
            revision=None,
            max_samples=max_samples,
            seed=trainer.seed,
            cache_root=trainer.cache_root,
        )

    prepared = prepare_split(
        split=split,
        dataset_name=DATASET_ID,
        revision=None,
        max_samples=max_samples,
        seed=trainer.seed,
        epoch=epoch_index,
        training=training,
        cache_root=trainer.cache_root,
        streaming=trainer.streaming,
    )

    return build_dataloader(
        prepared,
        batch_size=trainer.batch_size,
        num_workers=trainer.num_workers,
        training=training,
        seed=trainer.seed,
        epoch=epoch_index,
        include_metadata=trainer.include_metadata,
        pin_memory=trainer.pin_memory,
        multiprocessing_context=trainer.multiprocessing_context,
        persistent_workers=trainer.persistent_workers,
        prefetch_factor=trainer.prefetch_factor,
        drop_last=trainer.drop_last,
        crop_size=trainer.train_crop_size if training else None,
        crop_mode="random" if training and trainer.train_crop_size is not None else "none",
        random_flip=trainer.train_random_flip if training else False,
        random_rot90=trainer.train_random_rot90 if training else False,
    )


def normalize_example_splits(example_splits: Sequence[str] | str | None) -> list[str]:
    if example_splits is None:
        requested_splits = ["test"]
    elif isinstance(example_splits, str):
        requested_splits = [example_splits]
    else:
        requested_splits = list(example_splits)

    normalized: list[str] = []
    seen: set[str] = set()
    invalid: list[str] = []
    for split in requested_splits:
        if split not in EXAMPLE_SPLITS:
            invalid.append(str(split))
            continue
        if split in seen:
            continue
        seen.add(split)
        normalized.append(split)

    if invalid:
        supported = ", ".join(EXAMPLE_SPLITS)
        invalid_text = ", ".join(invalid)
        raise ValueError(f"example_splits must contain only {supported}; got: {invalid_text}")

    return normalized


def build_example_max_samples_by_split(
    *,
    train_max_samples: int | None,
    val_max_samples: int | None,
    test_max_samples: int | None,
) -> dict[str, int | None]:
    return {
        "train": train_max_samples,
        "validation": val_max_samples,
        "test": test_max_samples,
    }


def select_rgb(image: torch.Tensor) -> torch.Tensor:
    # 13채널 optical image에서 사람이 직관적으로 보는 RGB만 뽑는다.
    return image[list(RGB_CHANNELS)]


def normalize_rgb_triplet(
    cloudy: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 세 이미지는 같은 스케일로 보여야 비교가 쉽다.
    # 다만 prediction에 큰 이상치가 끼면 기준 스케일까지 깨져서
    # 정상인 cloudy/target 색이 같이 죽을 수 있으므로, 기준 범위는 입력/GT로만 잡는다.
    rgb_tensors = [select_rgb(tensor).detach().cpu().float() for tensor in (cloudy, prediction, target)]
    reference_tensors = [rgb_tensors[0], rgb_tensors[2]]
    merged = np.concatenate(
        [tensor.permute(1, 2, 0).reshape(-1, 3).numpy() for tensor in reference_tensors],
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
    # SAR 평균 맵이나 오차 맵은 단일 채널이므로 분위수 기반으로 0~1에 맞춰 보여 준다.
    array = image.detach().cpu().float().numpy()
    low, high = np.quantile(array, [0.02, 0.98])
    return np.clip((array - low) / max(high - low, 1e-6), 0.0, 1.0)


def _metadata_value(metadata: Mapping[str, Any], key: str, index: int) -> str:
    values = metadata.get(key, [])
    if not isinstance(values, (list, tuple)) or index >= len(values):
        return ""
    return str(values[index])


def _build_example_title(
    split_label: str,
    *,
    example_index: int,
    metadata: Mapping[str, Any],
    batch_index: int,
) -> str:
    season = _metadata_value(metadata, "season", batch_index)
    scene = _metadata_value(metadata, "scene", batch_index)
    patch = _metadata_value(metadata, "patch", batch_index)
    return f"{split_label} example {example_index} | {season}/scene_{scene}/patch_{patch}"


def _build_example_panels(
    *,
    cloudy: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    sar: torch.Tensor,
) -> tuple[tuple[str, np.ndarray, str | None], ...]:
    cloudy_rgb, prediction_rgb, target_rgb = normalize_rgb_triplet(cloudy, prediction, target)
    sar_map = normalize_map(sar.mean(dim=0))
    error_map = normalize_map((prediction - target).abs().mean(dim=0))
    return (
        ("Cloudy RGB", cloudy_rgb, None),
        ("Prediction RGB", prediction_rgb, None),
        ("Target RGB", target_rgb, None),
        ("SAR Mean", sar_map, "gray"),
        ("Abs Error", error_map, "magma"),
    )


def _save_example_figure(
    *,
    output_dir: Path,
    split_label: str,
    example_index: int,
    title: str,
    panels: Sequence[tuple[str, np.ndarray, str | None]],
) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle(title, fontsize=11)
    for ax, (panel_title, image, cmap) in zip(axes, panels):
        if cmap is None:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap=cmap)
        ax.set_title(panel_title)
        ax.axis("off")

    fig.tight_layout()
    fig.subplots_adjust(top=0.80)

    path = output_dir / f"{split_label}_example_{example_index:02d}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _render_restoration_examples(
    trainer: Trainer,
    dataloader,
    *,
    output_dir: Path,
    num_examples: int,
    split_label: str,
) -> list[Path]:
    # 학습이 끝난 뒤 실제 복원 품질을 빠르게 확인할 수 있도록
    # cloudy / prediction / target / SAR / error를 한 장에 묶어서 저장한다.
    if num_examples <= 0:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    was_training = trainer.model.training
    trainer.model.eval()

    saved_paths: list[Path] = []
    iterator = iter(dataloader)
    try:
        with torch.no_grad():
            while len(saved_paths) < num_examples:
                try:
                    batch = next(iterator)
                except StopIteration:
                    break

                sar = batch["sar"]
                cloudy = batch["cloudy"]
                target = batch["target"]
                metadata = batch.get("meta", {})
                prediction = trainer.predict(batch)
                if not isinstance(prediction, torch.Tensor):
                    raise TypeError("trainer.predict(batch) must return a torch.Tensor")
                prediction = prediction.detach().cpu()

                for index in range(prediction.shape[0]):
                    example_index = len(saved_paths) + 1
                    panels = _build_example_panels(
                        cloudy=cloudy[index],
                        prediction=prediction[index],
                        target=target[index],
                        sar=sar[index],
                    )
                    title = _build_example_title(
                        split_label,
                        example_index=example_index,
                        metadata=metadata,
                        batch_index=index,
                    )
                    saved_paths.append(
                        _save_example_figure(
                            output_dir=output_dir,
                            split_label=split_label,
                            example_index=example_index,
                            title=title,
                            panels=panels,
                        )
                    )

                    if len(saved_paths) == num_examples:
                        break
    finally:
        # iterator를 빨리 정리해 두면 worker가 떠 있는 상태로 오래 남지 않는다.
        del iterator
        trainer.model.train(was_training)

    print(f"saved examples: {output_dir}")
    return saved_paths


def _validate_example_split(split: str) -> str:
    if split not in EXAMPLE_SPLITS:
        supported = ", ".join(EXAMPLE_SPLITS)
        raise ValueError(f"split must be one of {supported}; got: {split}")
    return split


def _resolve_example_request(
    trainer: Trainer,
    dataloader,
    *,
    split: str | None,
    max_samples: int | None,
    epoch: int | None,
    stage: str | None,
) -> tuple[Any, str]:
    split_label = stage
    if split is not None:
        split_label = _validate_example_split(split)

    if dataloader is not None:
        if split_label is None:
            raise TypeError("save_restoration_examples() requires either split or stage when dataloader is provided")
        return dataloader, str(split_label)

    if split is None or epoch is None:
        raise TypeError("save_restoration_examples() requires split and epoch when dataloader is not provided")

    return (
        build_loader(
            trainer,
            split=split,
            max_samples=max_samples,
            training=False,
            epoch_index=max(epoch - 1, 0),
        ),
        split,
    )


def save_restoration_examples(
    trainer: Trainer,
    dataloader=None,
    *,
    output_dir: Path,
    num_examples: int,
    split: str | None = None,
    max_samples: int | None = None,
    epoch: int | None = None,
    stage: str | None = None,
) -> list[Path]:
    # 공용 러너에서는 split/epoch만 넘기면 loader 생성부터 저장까지 한 번에 처리한다.
    # dataloader 직접 주입은 notebook preview 호환을 위한 fallback이다.
    if num_examples <= 0:
        return []

    dataloader, split_label = _resolve_example_request(
        trainer,
        dataloader,
        split=split,
        max_samples=max_samples,
        epoch=epoch,
        stage=stage,
    )

    return _render_restoration_examples(
        trainer=trainer,
        dataloader=dataloader,
        output_dir=output_dir,
        num_examples=num_examples,
        split_label=str(split_label),
    )


def save_examples_for_splits(
    trainer: Trainer,
    *,
    splits: Sequence[str],
    max_samples_by_split: Mapping[str, int | None],
    epoch: int,
    output_dir: Path,
    num_examples: int,
) -> dict[str, list[Path]]:
    if num_examples <= 0 or not splits:
        return {}

    saved_paths_by_split: dict[str, list[Path]] = {}
    for split in splits:
        saved_paths_by_split[split] = save_restoration_examples(
            trainer,
            split=split,
            max_samples=max_samples_by_split[split],
            epoch=epoch,
            output_dir=output_dir / split,
            num_examples=num_examples,
        )
    return saved_paths_by_split


def build_example_save_config(
    *,
    train_max_samples: int | None,
    val_max_samples: int | None,
    test_max_samples: int | None,
    num_examples: int,
    example_splits: Sequence[str] | str | None,
) -> ExampleSaveConfig:
    return ExampleSaveConfig(
        splits=tuple(normalize_example_splits(example_splits)),
        max_samples_by_split=build_example_max_samples_by_split(
            train_max_samples=train_max_samples,
            val_max_samples=val_max_samples,
            test_max_samples=test_max_samples,
        ),
        num_examples=num_examples,
    )


def examples_epoch_dir(output_dir: Path, *, epoch: int) -> Path:
    return output_dir / "examples" / f"epoch_{epoch:03d}"


def should_save_periodic_examples(epoch: int, *, save_every_n_epochs: int) -> bool:
    if save_every_n_epochs <= 0:
        return False
    if epoch <= 0:
        return False
    return epoch % save_every_n_epochs == 0


def maybe_save_epoch_examples(
    trainer: Trainer,
    *,
    output_dir: Path,
    epoch: int,
    example_config: ExampleSaveConfig,
    saved_epochs: set[int],
) -> dict[str, list[Path]]:
    if epoch in saved_epochs or example_config.num_examples <= 0 or not example_config.splits:
        return {}

    saved_paths_by_split = save_examples_for_splits(
        trainer,
        splits=example_config.splits,
        max_samples_by_split=example_config.max_samples_by_split,
        epoch=epoch,
        output_dir=examples_epoch_dir(output_dir, epoch=epoch),
        num_examples=example_config.num_examples,
    )
    saved_epochs.add(epoch)
    return saved_paths_by_split


def should_save_examples_for_epoch(
    *,
    example_mode: Literal["best", "after_test"],
    best_improved: bool,
    periodic_due: bool,
) -> bool:
    if periodic_due:
        return True
    return example_mode == "best" and best_improved


def maybe_update_best_state(
    trainer: Trainer,
    record: StepRecord,
    *,
    output_dir: Path,
    best_state: BestState | None,
    selector: BestEpochSelector,
) -> tuple[BestState | None, bool]:
    epoch = int(record["epoch"])
    score = score_epoch(record, selector=selector)
    if not is_better_score(
        score,
        None if best_state is None else best_state.score,
        mode=selector.mode,
    ):
        return best_state, False

    checkpoint_path = trainer.save_checkpoint(best_checkpoint_path(output_dir))
    updated_best_state = save_best_state(
        output_dir,
        epoch=epoch,
        score=score,
        selector=selector,
        checkpoint_path=checkpoint_path,
    )
    return updated_best_state, True


def maybe_save_epoch_checkpoint(trainer: Trainer, *, save_every_n_epochs: int) -> Path | None:
    if save_every_n_epochs <= 0:
        return None
    if trainer.current_epoch <= 0 or trainer.current_epoch % save_every_n_epochs != 0:
        return None

    checkpoint_path = trainer.save_checkpoint()
    print(f"saved checkpoint: {checkpoint_path}")
    return checkpoint_path


def run_test_and_record(
    trainer: Trainer,
    history: list[dict[str, int | float]],
    *,
    global_step: int,
) -> dict[str, Any]:
    test_record = trainer.test()
    append_history(
        history,
        {
            "epoch": trainer.current_epoch,
            "test": test_record,
        },
        global_step=global_step,
    )
    return test_record


def validate_main_args(
    *,
    output_dir: str | Path,
    resume: str | Path | None,
    example_mode: Literal["best", "after_test"],
    save_every_n_epochs: int,
) -> tuple[Path, Path | None]:
    resolved_output_dir = Path(output_dir)
    resolved_resume = Path(resume) if resume is not None else None

    if example_mode not in {"best", "after_test"}:
        raise ValueError("example_mode must be either 'best' or 'after_test'")
    if save_every_n_epochs < 0:
        raise ValueError("save_every_n_epochs must be greater than or equal to zero")

    return resolved_output_dir, resolved_resume


def normalize_data_loading_config(
    *,
    streaming: bool,
    train_max_samples: int | None,
    val_max_samples: int | None,
    test_max_samples: int | None,
) -> tuple[int | None, int | None, int | None]:
    if streaming:
        return train_max_samples, val_max_samples, test_max_samples

    ignored_limits = {
        name: value
        for name, value in (
            ("train_max_samples", train_max_samples),
            ("val_max_samples", val_max_samples),
            ("test_max_samples", test_max_samples),
        )
        if value is not None
    }
    if ignored_limits:
        ignored_text = ", ".join(f"{name}={value}" for name, value in ignored_limits.items())
        print(f"streaming=false uses the full dataset; ignoring sample limits: {ignored_text}")
    return None, None, None


def load_training_state(
    trainer: Trainer,
    *,
    output_dir: Path,
    resume: Path | None,
    selector: BestEpochSelector,
) -> BestState | None:
    best_state = load_best_state(output_dir, selector=selector) if resume is not None else None
    if resume is not None:
        trainer.load_checkpoint(resume)
    return best_state


def handle_training_epoch(
    trainer: Trainer,
    record: StepRecord,
    *,
    history: list[dict[str, int | float]],
    output_dir: Path,
    best_state: BestState | None,
    best_selector: BestEpochSelector,
    example_mode: Literal["best", "after_test"],
    example_config: ExampleSaveConfig,
    save_every_n_epochs: int,
    saved_example_epochs: set[int],
) -> BestState | None:
    epoch = int(record["epoch"])
    append_history(history, record, global_step=trainer.global_step)
    maybe_save_epoch_checkpoint(trainer, save_every_n_epochs=save_every_n_epochs)

    next_best_state, best_improved = maybe_update_best_state(
        trainer,
        record,
        output_dir=output_dir,
        best_state=best_state,
        selector=best_selector,
    )
    periodic_due = should_save_periodic_examples(epoch, save_every_n_epochs=save_every_n_epochs)
    if should_save_examples_for_epoch(
        example_mode=example_mode,
        best_improved=best_improved,
        periodic_due=periodic_due,
    ):
        maybe_save_epoch_examples(
            trainer,
            output_dir=output_dir,
            epoch=epoch,
            example_config=example_config,
            saved_epochs=saved_example_epochs,
        )

    return next_best_state


def run_training_loop(
    trainer: Trainer,
    *,
    output_dir: Path,
    best_state: BestState | None,
    best_selector: BestEpochSelector,
    example_mode: Literal["best", "after_test"],
    example_config: ExampleSaveConfig,
    save_every_n_epochs: int,
) -> tuple[list[dict[str, int | float]], set[int]]:
    history: list[dict[str, int | float]] = []
    saved_example_epochs: set[int] = set()
    current_best_state = best_state

    while trainer.current_epoch < trainer.epochs:
        record = trainer.step()
        current_best_state = handle_training_epoch(
            trainer,
            record,
            history=history,
            output_dir=output_dir,
            best_state=current_best_state,
            best_selector=best_selector,
            example_mode=example_mode,
            example_config=example_config,
            save_every_n_epochs=save_every_n_epochs,
            saved_example_epochs=saved_example_epochs,
        )

    return history, saved_example_epochs


def finalize_after_training(
    trainer: Trainer,
    history: list[dict[str, int | float]],
    *,
    output_dir: Path,
    run_test: bool,
    example_mode: Literal["best", "after_test"],
    example_config: ExampleSaveConfig,
    saved_example_epochs: set[int],
) -> None:
    if example_mode == "after_test":
        run_test_and_record(trainer, history, global_step=trainer.global_step)
        maybe_save_epoch_examples(
            trainer,
            output_dir=output_dir,
            epoch=max(trainer.current_epoch, 1),
            example_config=example_config,
            saved_epochs=saved_example_epochs,
        )
    elif run_test:
        run_test_and_record(trainer, history, global_step=trainer.global_step)

    save_history_plot(history, output_dir / "history.png")


def main(
    *,
    batch_size: int = 8,
    accum_steps: int = 1,
    seed: int = 42,
    max_epochs: int = 10,
    train_max_samples: int | None = 16384,
    val_max_samples: int | None = 2048,
    test_max_samples: int | None = 2048,
    streaming: bool = True,
    output_dir: str | Path = Path("artifacts"),
    cache_dir: str | Path | None = None,
    resume: str | Path | None = None,
    num_workers: int | str = "auto",
    multiprocessing_context: str | None = None,
    scheduler_timing: SchedulerTiming = "after_validation",
    train_crop_size: int | None = 128,
    train_random_flip: bool = True,
    train_random_rot90: bool = True,
    save_every_n_epochs: int = 0,
    run_test: bool = True,
    num_examples: int = 4,
    example_mode: Literal["best", "after_test"] = "best",
    example_splits: list[str] | None = None,
) -> None:
    # main은 "모델/옵티마이저 조립 -> Trainer 실행 -> 결과물 저장"만 담당한다.
    # Trainer가 이미 해 주는 일은 최대한 그대로 맡기고, 여기서는 프로젝트 전용 후처리만 남긴다.
    # Trainer 생성은 build_trainer()로 모으고, main은 실행 순서와 산출물 정책만 관리한다.
    output_dir, resume = validate_main_args(
        output_dir=output_dir,
        resume=resume,
        example_mode=example_mode,
        save_every_n_epochs=save_every_n_epochs,
    )
    train_max_samples, val_max_samples, test_max_samples = normalize_data_loading_config(
        streaming=streaming,
        train_max_samples=train_max_samples,
        val_max_samples=val_max_samples,
        test_max_samples=test_max_samples,
    )
    example_config = build_example_save_config(
        train_max_samples=train_max_samples,
        val_max_samples=val_max_samples,
        test_max_samples=test_max_samples,
        num_examples=num_examples,
        example_splits=example_splits,
    )

    trainer = build_trainer(
        batch_size=batch_size,
        accum_steps=accum_steps,
        seed=seed,
        max_epochs=max_epochs,
        train_max_samples=train_max_samples,
        val_max_samples=val_max_samples,
        test_max_samples=test_max_samples,
        streaming=streaming,
        output_dir=output_dir,
        cache_dir=cache_dir,
        num_workers=num_workers,
        multiprocessing_context=multiprocessing_context,
        scheduler_timing=scheduler_timing,
        train_crop_size=train_crop_size,
        train_random_flip=train_random_flip,
        train_random_rot90=train_random_rot90,
    )
    best_selector = build_best_epoch_selector()
    best_state = load_training_state(
        trainer,
        output_dir=output_dir,
        resume=resume,
        selector=best_selector,
    )
    history, saved_example_epochs = run_training_loop(
        trainer,
        output_dir=output_dir,
        best_state=best_state,
        best_selector=best_selector,
        example_mode=example_mode,
        example_config=example_config,
        save_every_n_epochs=save_every_n_epochs,
    )
    finalize_after_training(
        trainer,
        history,
        output_dir=output_dir,
        run_test=run_test,
        example_mode=example_mode,
        example_config=example_config,
        saved_example_epochs=saved_example_epochs,
    )


if __name__ == "__main__":
    main()
