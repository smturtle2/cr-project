from __future__ import annotations

import json
import math
import random
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from huggingface_hub import get_token
from torch import nn

from cr_train import Trainer
from cr_train.data import DATASET_ID, build_dataloader
from cr_train.data.dataset import prepare_split
from cr_train.data.runtime import ensure_split_cache


# Sentinel-2 13채널 중 사람이 보기 쉬운 RGB 조합(B4, B3, B2) 인덱스다.
RGB_CHANNELS = (3, 2, 1)

# Trainer가 기대하는 `(prediction, batch)` 계약을 타입으로도 분명히 적어 둔다.
Batch = dict[str, Any]
LossFn = Callable[[torch.Tensor, Batch], torch.Tensor]
MetricFn = Callable[[torch.Tensor, Batch], torch.Tensor]
StepRecord = Mapping[str, Any]


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


def load_checkpoint(trainer: Trainer, checkpoint_path: Path, *, device: torch.device) -> None:
    # 최신 cr-train은 공개 resume API를 제공하지 않으므로
    # 저장 포맷(`model`, `optimizer`, `epoch`, `global_step`)에 맞춰 직접 복원한다.
    # PyTorch 버전에 따라 `weights_only` 인자 지원 여부가 달라서 여기서만 분기 처리한다.
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    # 분산 학습에서 model이 DDP로 감싸졌더라도 실제 state_dict 대상은 내부 module이다.
    model_owner = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    model_owner.load_state_dict(checkpoint["model"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer"])
    trainer.current_epoch = int(checkpoint["epoch"])
    trainer.global_step = int(checkpoint["global_step"])

    print(
        "resumed checkpoint:",
        f"path={checkpoint_path}",
        f"epoch={trainer.current_epoch}",
        f"global_step={trainer.global_step}",
    )


def flatten_record(record: Mapping[str, Any], *, global_step: int) -> dict[str, int | float]:
    # epoch 단위 결과를 plot용 row 하나로 바꾼다.
    # Trainer가 이미 사람이 읽기 좋은 요약을 출력하므로, 여기서는 시각화에 필요한 값만 남긴다.
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

        if "loss" in summary:
            row[f"{stage}_loss"] = float(summary["loss"])

        for name, value in summary.get("metrics", {}).items():
            row[f"{stage}_{name}"] = float(value)

    return row


def append_history(
    history: list[dict[str, int | float]],
    record: Mapping[str, Any],
    *,
    global_step: int,
) -> None:
    history.append(flatten_record(record, global_step=global_step))


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


def load_best_state(output_dir: Path, *, selector: BestEpochSelector) -> BestState | None:
    metadata_path = best_metadata_path(output_dir)
    if not metadata_path.exists():
        return None

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    selector_name = str(payload["selector_name"])
    selector_mode = str(payload["selector_mode"])
    if selector_name != selector.name or selector_mode != selector.mode:
        print(
            "ignored existing best metadata:",
            f"path={metadata_path}",
            f"selector={selector_name}/{selector_mode}",
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
    source_checkpoint_path: Path,
) -> BestState:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = best_checkpoint_path(output_dir)
    shutil.copy2(source_checkpoint_path, best_path)

    payload = {
        "epoch": int(epoch),
        "score": float(score),
        "checkpoint_path": str(best_path),
        "selector_name": selector.name,
        "selector_mode": selector.mode,
        "source_checkpoint_path": str(source_checkpoint_path),
    }
    best_metadata_path(output_dir).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "updated best checkpoint:",
        f"epoch={epoch}",
        f"{selector.name}={score:.6f}",
        f"path={best_path}",
    )
    return BestState(
        epoch=int(epoch),
        score=float(score),
    )


def remove_checkpoint_file(checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        return

    checkpoint_path.unlink()
    print(f"removed checkpoint: {checkpoint_path}")


def save_history_plot(history: list[dict[str, int | float]], path: Path) -> None:
    # 학습 결과를 한 장으로 남기기 위해 loss, metric, epoch 시간대를 분리해서 그린다.
    # 파일 기반 결과물이 있으면 notebook 없이도 학습 흐름을 다시 확인하기 쉽다.
    if not history:
        return

    import matplotlib.pyplot as plt

    metric_keys = sorted(
        {key for row in history for key in row if key not in {"epoch", "global_step"}}
    )
    loss_keys = [key for key in metric_keys if key.endswith("_loss")]
    time_keys = [key for key in metric_keys if key == "elapsed_sec"]
    other_keys = [key for key in metric_keys if key not in loss_keys and key not in time_keys]
    groups = [keys for keys in (loss_keys, other_keys, time_keys) if keys]
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
        if keys == time_keys:
            ax.set_ylabel("seconds")
        else:
            ax.set_ylabel("value")

    axes[0].set_title("training and evaluation history")
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved plot: {path}")


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
        persistent_workers=trainer.persistent_workers,
        prefetch_factor=trainer.prefetch_factor,
        drop_last=trainer.drop_last if training else False,
    )


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


def save_restoration_examples(
    model: nn.Module,
    dataloader,
    *,
    device: torch.device,
    output_dir: Path,
    num_examples: int,
    stage: str,
) -> list[Path]:
    # 학습이 끝난 뒤 실제 복원 품질을 빠르게 확인할 수 있도록
    # cloudy / prediction / target / SAR / error를 한 장에 묶어서 저장한다.
    if num_examples <= 0:
        return []

    import matplotlib.pyplot as plt

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

                sar = batch["sar"]
                cloudy = batch["cloudy"]
                target = batch["target"]
                metadata = batch.get("meta", {})
                prediction = model(sar.to(device), cloudy.to(device)).cpu()

                for index in range(prediction.shape[0]):
                    cloudy_rgb, prediction_rgb, target_rgb = normalize_rgb_triplet(
                        cloudy[index],
                        prediction[index],
                        target[index],
                    )
                    sar_map = normalize_map(sar[index].mean(dim=0))
                    error_map = normalize_map((prediction[index] - target[index]).abs().mean(dim=0))

                    # metadata는 batch 단위 list라서 index별 값을 직접 안전하게 꺼낸다.
                    seasons = metadata.get("season", [])
                    scenes = metadata.get("scene", [])
                    patches = metadata.get("patch", [])
                    season = str(seasons[index]) if isinstance(seasons, (list, tuple)) and index < len(seasons) else ""
                    scene = str(scenes[index]) if isinstance(scenes, (list, tuple)) and index < len(scenes) else ""
                    patch = str(patches[index]) if isinstance(patches, (list, tuple)) and index < len(patches) else ""

                    title = (
                        f"{stage} example {len(saved_paths) + 1} | "
                        f"{season}/scene_{scene}/patch_{patch}"
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
        # iterator를 빨리 정리해 두면 worker가 떠 있는 상태로 오래 남지 않는다.
        del iterator
        model.train(was_training)

    print(f"saved examples: {output_dir}")
    return saved_paths


def save_examples_for_epoch(
    trainer: Trainer,
    *,
    device: torch.device,
    split: str,
    max_samples: int | None,
    stage: str,
    epoch: int,
    output_dir: Path,
    num_examples: int,
) -> list[Path]:
    if num_examples <= 0:
        return []

    dataloader = build_loader(
        trainer,
        split=split,
        max_samples=max_samples,
        training=False,
        epoch_index=max(epoch - 1, 0),
    )
    return save_restoration_examples(
        model=trainer.model,
        dataloader=dataloader,
        device=device,
        output_dir=output_dir,
        num_examples=num_examples,
        stage=stage,
    )


def maybe_update_best_state(
    trainer: Trainer,
    record: StepRecord,
    *,
    output_dir: Path,
    device: torch.device,
    best_state: BestState | None,
    selector: BestEpochSelector,
    example_mode: Literal["best", "after_test"],
    val_max_samples: int | None,
    num_examples: int,
) -> BestState | None:
    epoch = int(record["epoch"])
    checkpoint_path = Path(str(record["checkpoint_path"]))
    score = score_epoch(record, selector=selector)
    if not is_better_score(
        score,
        None if best_state is None else best_state.score,
        mode=selector.mode,
    ):
        remove_checkpoint_file(checkpoint_path)
        return best_state

    updated_best_state = save_best_state(
        output_dir,
        epoch=epoch,
        score=score,
        selector=selector,
        source_checkpoint_path=checkpoint_path,
    )
    if example_mode == "best":
        save_examples_for_epoch(
            trainer,
            device=device,
            split="validation",
            max_samples=val_max_samples,
            stage="val",
            epoch=epoch,
            output_dir=output_dir / "examples" / "best" / f"epoch_{epoch:03d}",
            num_examples=num_examples,
        )
    remove_checkpoint_file(checkpoint_path)
    return updated_best_state


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


def main(
    *,
    batch_size: int = 8,
    seed: int = 42,
    max_epochs: int = 10,
    train_max_samples: int = 16384,
    val_max_samples: int = 2048,
    test_max_samples: int = 2048,
    output_dir: str | Path = Path("artifacts"),
    resume: str | Path | None = None,
    run_test: bool = True,
    num_examples: int = 4,
    example_mode: Literal["best", "after_test"] = "best",
) -> None:
    # main은 "모델/옵티마이저 조립 -> Trainer 실행 -> 결과물 저장"만 담당한다.
    # Trainer가 이미 해 주는 일은 최대한 그대로 맡기고, 여기서는 프로젝트 전용 후처리만 남긴다.
    # 그래서 Trainer 생성도 별도 래퍼 함수로 감싸지 않고 이 자리에서 바로 보이게 둔다.
    output_dir = Path(output_dir)
    resume = Path(resume) if resume is not None else None
    if example_mode not in {"best", "after_test"}:
        raise ValueError("example_mode must be either 'best' or 'after_test'")

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_hf_auth_status()

    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model().to(device)
    optimizer = build_optimizer(model)
    # 이 프로젝트가 Trainer에 어떤 값을 넘기는지 한눈에 보이도록 직접 생성한다.
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss=build_loss(),
        metrics=build_metrics(),
        max_train_samples=train_max_samples,
        max_val_samples=val_max_samples,
        max_test_samples=test_max_samples,
        output_dir=output_dir,
        batch_size=batch_size,
        epochs=max_epochs,
        seed=seed,
        train_crop_size=128,
        train_random_flip=True,
        train_random_rot90=True,
    )
    best_selector = build_best_epoch_selector()
    best_state = load_best_state(output_dir, selector=best_selector) if resume is not None else None

    if resume is not None:
        load_checkpoint(trainer, resume, device=device)

    history: list[dict[str, int | float]] = []
    while trainer.current_epoch < trainer.epochs:
        record = trainer.step()
        append_history(history, record, global_step=trainer.global_step)
        best_state = maybe_update_best_state(
            trainer,
            record,
            output_dir=output_dir,
            device=device,
            best_state=best_state,
            selector=best_selector,
            example_mode=example_mode,
            val_max_samples=val_max_samples,
            num_examples=num_examples,
        )

    if example_mode == "after_test":
        run_test_and_record(trainer, history, global_step=trainer.global_step)
        save_examples_for_epoch(
            trainer,
            device=device,
            split="test",
            max_samples=test_max_samples,
            stage="test",
            epoch=max(trainer.current_epoch, 1),
            output_dir=output_dir / "examples" / "test",
            num_examples=num_examples,
        )
    elif run_test:
        run_test_and_record(trainer, history, global_step=trainer.global_step)

    save_history_plot(history, output_dir / "history.png")


if __name__ == "__main__":
    main()
