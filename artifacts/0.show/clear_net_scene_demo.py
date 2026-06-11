#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import mimetypes
from pathlib import Path
import queue
import shutil
import sys
import threading
import time
from typing import Any, Mapping
import urllib.parse
import webbrowser

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from cr_train.data.dataset import decode_row
from cr_train.data.hf_v2 import (
    HFV2LocalBlockReader,
    HFV2StagedBlockReader,
    load_hf_v2_manifest,
    load_hf_v2_split_catalog,
)

from modules.model.CLEAR_Net import CLEAR_Net


SPLITS = ("train", "validation", "test")
SEASONS = ("spring", "summer", "fall", "winter")
PATCH_GRID_WIDTH = 29
SCENE_TILE_SIZE = 64
PATCH_STRIDE_FRACTION = 0.5
FEATHER_EDGE_WEIGHT = 0.15
RGB_CHANNELS = (3, 2, 1)
DEFAULT_DATASET_ROOT = Path("/dhdd/.cache/cr-train")
DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts/3.CLEAR-Net/v4/best.pt"
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".clear_net_scene_demo_cache"


class DemoError(ValueError):
    pass


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def safe_part(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def normalize_scene_value(scene: str) -> str:
    value = str(scene).strip()
    if value.startswith("scene_"):
        return value.split("scene_", 1)[1]
    return value


def scene_label(scene: str) -> str:
    return f"scene_{normalize_scene_value(scene)}"


def patch_grid_position(patch: str) -> tuple[int, int]:
    value = str(patch).strip().lower()
    if value.startswith("p") and value[1:].isdigit():
        patch_number = int(value[1:])
    elif value.isdigit():
        patch_number = int(value)
    else:
        raise DemoError(f"patch id is not numeric: {patch!r}")
    return patch_number // PATCH_GRID_WIDTH, patch_number % PATCH_GRID_WIDTH


def feather_weight(tile_size: int) -> np.ndarray:
    phase = np.linspace(0.0, np.pi, tile_size, dtype=np.float32)
    one_dim = FEATHER_EDGE_WEIGHT + (1.0 - FEATHER_EDGE_WEIGHT) * np.sin(phase)
    return (one_dim[:, None] * one_dim[None, :])[:, :, None]


def optical_rgb(image_chw: np.ndarray) -> np.ndarray:
    image = np.asarray(image_chw, dtype=np.float32)
    rgb = np.transpose(image[list(RGB_CHANNELS)], (1, 2, 0))
    rgb = np.clip(rgb / 5.0, 0.0, 1.0)
    return (rgb ** (1.0 / 2.2)).astype(np.float32)


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(image, dtype=np.float32) * 255.0, 0.0, 255.0).round().astype(np.uint8)


def save_rgb_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(image), mode="RGB").save(path)


def resize_tile(image: np.ndarray, tile_size: int) -> np.ndarray:
    if image.shape[0] == tile_size and image.shape[1] == tile_size:
        return image.astype(np.float32)
    pil = Image.fromarray(to_uint8(image), mode="RGB")
    pil = pil.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0


def ordered_blend_tile(
    canvas: np.ndarray,
    covered: np.ndarray,
    tile: np.ndarray,
    weight: np.ndarray,
    *,
    x: int,
    y: int,
) -> None:
    region = canvas[y : y + tile.shape[0], x : x + tile.shape[1]]
    coverage = covered[y : y + tile.shape[0], x : x + tile.shape[1]]
    alpha = weight.astype(np.float32)
    effective_alpha = np.where(coverage[:, :, None], alpha, 1.0)
    region[:] = tile * effective_alpha + region * (1.0 - effective_alpha)
    coverage[:] = True


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise DemoError("requested cuda but torch.cuda.is_available() is false")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_context(device: torch.device, mode: str):
    if mode == "off" or device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if mode == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def checkpoint_fingerprint(path: Path) -> str:
    stat = path.stat()
    stable = f"{path.resolve()}:{stat.st_size}:{int(stat.st_mtime)}"
    return hashlib.sha1(stable.encode("utf-8")).hexdigest()[:12]


class ClearNetPredictor:
    def __init__(
        self,
        *,
        checkpoint_path: Path,
        device: torch.device,
        mixed_precision: str,
        batch_size: int,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.mixed_precision = mixed_precision
        self.batch_size = int(batch_size)
        self.model: CLEAR_Net | None = None
        self.model_meta: dict[str, Any] | None = None
        self.lock = threading.Lock()

    def load_model(self) -> CLEAR_Net:
        if self.model is not None:
            return self.model
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
            raise DemoError("checkpoint must contain a model state dict under 'model'")

        model = CLEAR_Net(return_decomposition=True)
        load_result = model.load_state_dict(checkpoint["model"], strict=True)
        model.to(self.device).eval()
        self.model = model
        self.model_meta = {
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "global_step": int(checkpoint.get("global_step", -1)),
            "model_constructor": "CLEAR_Net(return_decomposition=True)",
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
            "device": str(self.device),
            "mixed_precision": self.mixed_precision,
            "batch_size": self.batch_size,
        }
        return model

    def predict(self, decoded_rows: list[dict[str, Any]]) -> list[np.ndarray]:
        return [prediction for _item, prediction in self.iter_predictions(decoded_rows)]

    def iter_predictions(self, decoded_rows: list[dict[str, Any]]):
        with self.lock:
            model = self.load_model()
            batch_size = self.batch_size
            start = 0
            with torch.inference_mode():
                while start < len(decoded_rows):
                    current_size = min(batch_size, len(decoded_rows) - start)
                    while True:
                        chunk = decoded_rows[start : start + current_size]
                        try:
                            sar = torch.from_numpy(np.stack([item["sar"] for item in chunk], axis=0)).to(self.device)
                            cloudy = torch.from_numpy(np.stack([item["cloudy"] for item in chunk], axis=0)).to(self.device)
                            with autocast_context(self.device, self.mixed_precision):
                                output = model(sar, cloudy)
                            prediction = output["prediction"] if isinstance(output, Mapping) else output
                            prediction_arrays = [item.detach().cpu().float().numpy() for item in prediction]
                            chunk_items = list(chunk)
                            start += current_size
                            batch_size = min(batch_size, current_size)
                            for item, prediction_array in zip(chunk_items, prediction_arrays, strict=True):
                                yield item, prediction_array
                            break
                        except RuntimeError as exc:
                            is_cuda_oom = self.device.type == "cuda" and "out of memory" in str(exc).lower()
                            if not is_cuda_oom or current_size <= 1:
                                raise
                            current_size = max(1, current_size // 2)
                            torch.cuda.empty_cache()


class SceneCatalog:
    def __init__(self, *, streaming: bool, dataset_root: Path | None) -> None:
        self.streaming = bool(streaming)
        self.dataset_root = dataset_root
        self.manifest = load_hf_v2_manifest(dataset_root=dataset_root, streaming=streaming)
        self.catalogs = {
            split: load_hf_v2_split_catalog(
                split=split,
                dataset_root=dataset_root,
                streaming=streaming,
            )
            for split in SPLITS
        }
        self.scenes = self._build_scenes()

    def _build_scenes(self) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
        scenes: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
            split: {season: {} for season in SEASONS}
            for split in SPLITS
        }
        for split, catalog in self.catalogs.items():
            for block in catalog.get("blocks", []):
                season = str(block.get("season", ""))
                scene = str(block.get("scene", "")).removeprefix("scene_")
                if season not in SEASONS or not scene:
                    continue
                item = scenes[split][season].setdefault(
                    scene,
                    {
                        "split": split,
                        "season": season,
                        "scene": scene,
                        "scene_label": f"scene_{scene}",
                        "rows": 0,
                        "blocks": 0,
                    },
                )
                item["rows"] += int(block["row_count"])
                item["blocks"] += 1
        return scenes

    def scene_blocks(self, *, split: str, season: str, scene: str) -> list[dict[str, Any]]:
        if split not in SPLITS:
            raise DemoError(f"unsupported split: {split!r}")
        if season not in SEASONS:
            raise DemoError(f"unsupported season: {season!r}")
        scene_value = normalize_scene_value(scene)
        target_label = f"scene_{scene_value}"
        blocks = [
            block
            for block in self.catalogs[split].get("blocks", [])
            if str(block.get("season")) == season and str(block.get("scene")) == target_label
        ]
        if not blocks:
            raise DemoError(f"scene not found: {split}/{season}/{target_label}")
        return blocks

    def build_queue(
        self,
        *,
        split: str,
        season: str,
        scenes: tuple[str, ...] | None,
        max_scenes: int,
    ) -> list[dict[str, Any]]:
        items = list(self.scenes[split][season].values())
        items.sort(
            key=lambda item: (
                0,
                int(item["scene"]),
            )
            if str(item["scene"]).isdigit()
            else (1, str(item["scene"]))
        )
        if scenes:
            requested = {normalize_scene_value(scene) for scene in scenes}
            items = [item for item in items if str(item["scene"]) in requested]
        if max_scenes > 0:
            items = items[:max_scenes]
        if not items:
            raise DemoError(f"no scenes available for {split}/{season}")
        return [
            {
                "index": index,
                "split": item["split"],
                "season": item["season"],
                "scene": item["scene"],
                "scene_label": item["scene_label"],
                "rows": int(item["rows"]),
                "blocks": int(item["blocks"]),
            }
            for index, item in enumerate(items)
        ]

    def build_scene_options(
        self,
        *,
        max_scenes: int,
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        return {
            split: {
                season: self.build_queue(
                    split=split,
                    season=season,
                    scenes=None,
                    max_scenes=max_scenes,
                )
                for season in SEASONS
                if self.scenes[split][season]
            }
            for split in SPLITS
        }


class SceneRenderer:
    def __init__(
        self,
        *,
        catalog: SceneCatalog,
        predictor: ClearNetPredictor,
        cache_dir: Path,
        tile_size: int,
        streaming: bool,
        dataset_root: Path | None,
        checkpoint_hash: str,
    ) -> None:
        self.catalog = catalog
        self.predictor = predictor
        self.cache_dir = cache_dir
        self.tile_size = int(tile_size)
        self.stride = max(1, int(round(self.tile_size * PATCH_STRIDE_FRACTION)))
        self.streaming = bool(streaming)
        self.dataset_root = dataset_root
        self.checkpoint_hash = checkpoint_hash

    def render_id(self, *, split: str, season: str, scene: str) -> str:
        scene_value = normalize_scene_value(scene)
        stable = f"{split}:{season}:{scene_value}:tile{self.tile_size}:ckpt{self.checkpoint_hash}:demo"
        digest = hashlib.sha1(stable.encode("utf-8")).hexdigest()[:12]
        return f"{safe_part(split)}_{safe_part(season)}_scene_{safe_part(scene_value)}_{digest}"

    def render_dir(self, render_id: str) -> Path:
        return self.cache_dir / "renders" / render_id

    def state_path(self, render_id: str) -> Path:
        return self.render_dir(render_id) / "state.json"

    def cached_state(self, render_id: str, *, require_prediction: bool) -> dict[str, Any] | None:
        state_path = self.state_path(render_id)
        cloudy_path = self.render_dir(render_id) / "cloudy.png"
        prediction_path = self.render_dir(render_id) / "prediction.png"
        if not state_path.is_file() or not cloudy_path.is_file():
            return None
        if require_prediction and not prediction_path.is_file():
            return None
        state = read_json(state_path)
        if require_prediction:
            state["has_prediction"] = True
            state["active_layer"] = "prediction"
        return state

    def render_scene(
        self,
        *,
        split: str,
        season: str,
        scene: str,
        infer: bool,
    ) -> dict[str, Any]:
        render_id = self.render_id(split=split, season=season, scene=scene)
        cached = self.cached_state(render_id, require_prediction=infer)
        if cached is not None:
            return cached

        start = time.perf_counter()
        blocks = self.catalog.scene_blocks(split=split, season=season, scene=scene)
        temp_dir = self.render_dir(render_id).with_name(self.render_dir(render_id).name + ".tmp")
        final_dir = self.render_dir(render_id)
        previous_state = self.cached_state(render_id, require_prediction=False)
        if previous_state is not None and (final_dir / "cloudy.png").is_file():
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.copytree(final_dir, temp_dir)
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
            temp_dir.mkdir(parents=True, exist_ok=True)

        decoded_rows = self._load_decoded_rows(split=split, blocks=blocks)
        if not decoded_rows:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise DemoError("selected scene has no rows")

        layout = self._layout(decoded_rows)
        if previous_state is None or not (temp_dir / "cloudy.png").is_file():
            cloudy_mosaic = self._mosaic(
                decoded_rows,
                layout=layout,
                layer_name="cloudy",
            )
            save_rgb_png(temp_dir / "cloudy.png", cloudy_mosaic)

        has_prediction = bool(previous_state and previous_state.get("has_prediction"))
        if infer and not (temp_dir / "prediction.png").is_file():
            predictions = self.predictor.predict(decoded_rows)
            for item, prediction in zip(decoded_rows, predictions, strict=True):
                item["prediction"] = prediction
            prediction_mosaic = self._mosaic(
                decoded_rows,
                layout=layout,
                layer_name="prediction",
            )
            save_rgb_png(temp_dir / "prediction.png", prediction_mosaic)
            has_prediction = True

        state = self._state(
            render_id=render_id,
            split=split,
            season=season,
            scene=scene,
            layout=layout,
            decoded_rows=decoded_rows,
            rows=len(decoded_rows),
            blocks=len(blocks),
            seconds=time.perf_counter() - start,
            has_prediction=has_prediction,
            active_layer="prediction" if infer and has_prediction else "cloudy",
        )
        atomic_write_json(temp_dir / "state.json", state)
        shutil.rmtree(final_dir, ignore_errors=True)
        temp_dir.replace(final_dir)
        return state

    def stream_infer_scene(
        self,
        *,
        split: str,
        season: str,
        scene: str,
        send_event,
    ) -> dict[str, Any]:
        render_id = self.render_id(split=split, season=season, scene=scene)
        cached = self.cached_state(render_id, require_prediction=True)
        if cached is not None:
            send_event({"type": "cached", "render": cached})
            return cached

        started_at = time.perf_counter()
        send_event({"type": "status", "message": f"{scene_label(scene)} loading scene rows"})
        blocks = self.catalog.scene_blocks(split=split, season=season, scene=scene)
        final_dir = self.render_dir(render_id)
        final_dir.mkdir(parents=True, exist_ok=True)

        decoded_rows = self._load_decoded_rows(split=split, blocks=blocks)
        if not decoded_rows:
            raise DemoError("selected scene has no rows")

        layout = self._layout(decoded_rows)
        cloudy_path = final_dir / "cloudy.png"
        if not cloudy_path.is_file():
            cloudy_mosaic = self._mosaic(decoded_rows, layout=layout, layer_name="cloudy")
            save_rgb_png(cloudy_path, cloudy_mosaic)

        state = self._state(
            render_id=render_id,
            split=split,
            season=season,
            scene=scene,
            layout=layout,
            decoded_rows=decoded_rows,
            rows=len(decoded_rows),
            blocks=len(blocks),
            seconds=0.0,
            has_prediction=False,
            active_layer="cloudy",
        )
        atomic_write_json(final_dir / "state.json", state)
        send_event({"type": "start", "render": state, "total": len(decoded_rows)})

        tiles_dir = final_dir / "tiles"
        shutil.rmtree(tiles_dir, ignore_errors=True)
        tiles_dir.mkdir(parents=True, exist_ok=True)

        for index, (item, prediction) in enumerate(self.predictor.iter_predictions(decoded_rows), start=1):
            item["prediction"] = prediction
            patch_name = str(item["patch"])
            tile_path = tiles_dir / f"{safe_part(patch_name)}.png"
            save_rgb_png(tile_path, resize_tile(optical_rgb(prediction), self.tile_size))
            patch_payload = {
                "type": "patch",
                "index": index,
                "total": len(decoded_rows),
                "patch": patch_name,
                "grid_row": int(item["grid_row"]),
                "grid_col": int(item["grid_col"]),
                "x": int((int(item["grid_col"]) - layout["min_col"]) * self.stride),
                "y": int((int(item["grid_row"]) - layout["min_row"]) * self.stride),
                "width": self.tile_size,
                "height": self.tile_size,
                "render_width": int(layout["width"]),
                "render_height": int(layout["height"]),
                "tile_url": f"/asset/{render_id}/tiles/{safe_part(patch_name)}.png",
            }
            send_event(patch_payload)

        prediction_mosaic = self._mosaic(decoded_rows, layout=layout, layer_name="prediction")
        save_rgb_png(final_dir / "prediction.png", prediction_mosaic)
        final_state = self._state(
            render_id=render_id,
            split=split,
            season=season,
            scene=scene,
            layout=layout,
            decoded_rows=decoded_rows,
            rows=len(decoded_rows),
            blocks=len(blocks),
            seconds=time.perf_counter() - started_at,
            has_prediction=True,
            active_layer="prediction",
        )
        atomic_write_json(final_dir / "state.json", final_state)
        send_event({"type": "complete", "render": final_state})
        return final_state

    def _load_decoded_rows(self, *, split: str, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        decoded_rows: list[dict[str, Any]] = []
        if self.streaming:
            reader = HFV2StagedBlockReader(split=split)
            reader.prepare_blocks(tuple(blocks), worker_count=1)
        else:
            if self.dataset_root is None:
                raise DemoError("dataset_root is required when streaming is disabled")
            reader = HFV2LocalBlockReader(
                dataset_root=self.dataset_root,
                block_path_by_key={str(block["cache_key"]): str(block["path"]) for block in blocks},
                row_count_by_key={str(block["cache_key"]): int(block["row_count"]) for block in blocks},
            )
        try:
            for block in blocks:
                cache_key = str(block["cache_key"])
                block_rows = reader.load_block(cache_key)
                try:
                    for local_index, row in enumerate(block_rows):
                        decoded = decode_row(row, include_metadata=True)
                        patch = str(decoded["meta"]["patch"])
                        grid_row, grid_col = patch_grid_position(patch)
                        decoded_rows.append(
                            {
                                "local_index": local_index,
                                "block": block,
                                "patch": patch,
                                "grid_row": grid_row,
                                "grid_col": grid_col,
                                "sar": decoded["sar"],
                                "cloudy": decoded["cloudy"],
                            }
                        )
                finally:
                    release_block = getattr(reader, "release_block", None)
                    if release_block is not None:
                        release_block(cache_key)
        finally:
            close = getattr(reader, "close", None)
            if close is not None:
                close()

        decoded_rows.sort(key=lambda item: (int(item["grid_row"]), int(item["grid_col"]), str(item["patch"])))
        return decoded_rows

    def _layout(self, decoded_rows: list[dict[str, Any]]) -> dict[str, int]:
        min_row = min(int(item["grid_row"]) for item in decoded_rows)
        max_row = max(int(item["grid_row"]) for item in decoded_rows)
        min_col = min(int(item["grid_col"]) for item in decoded_rows)
        max_col = max(int(item["grid_col"]) for item in decoded_rows)
        grid_rows = max_row - min_row + 1
        grid_cols = max_col - min_col + 1
        return {
            "min_row": min_row,
            "max_row": max_row,
            "min_col": min_col,
            "max_col": max_col,
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "width": (grid_cols - 1) * self.stride + self.tile_size,
            "height": (grid_rows - 1) * self.stride + self.tile_size,
        }

    def _mosaic(
        self,
        decoded_rows: list[dict[str, Any]],
        *,
        layout: dict[str, int],
        layer_name: str,
    ) -> np.ndarray:
        accum = np.zeros((layout["height"], layout["width"], 3), dtype=np.float32)
        weights = np.zeros((layout["height"], layout["width"], 1), dtype=np.float32)
        tile_weight = feather_weight(self.tile_size)
        for item in decoded_rows:
            grid_row = int(item["grid_row"])
            grid_col = int(item["grid_col"])
            x = (grid_col - layout["min_col"]) * self.stride
            y = (grid_row - layout["min_row"]) * self.stride
            tile = resize_tile(optical_rgb(item[layer_name]), self.tile_size)
            accum[y : y + self.tile_size, x : x + self.tile_size] += tile * tile_weight
            weights[y : y + self.tile_size, x : x + self.tile_size] += tile_weight

        covered = weights[:, :, 0] > 0
        mosaic = np.full_like(accum, 0.92, dtype=np.float32)
        mosaic[covered] = accum[covered] / weights[covered]
        return mosaic

    def _state(
        self,
        *,
        render_id: str,
        split: str,
        season: str,
        scene: str,
        layout: dict[str, int],
        decoded_rows: list[dict[str, Any]],
        rows: int,
        blocks: int,
        seconds: float,
        has_prediction: bool,
        active_layer: str,
    ) -> dict[str, Any]:
        return {
            "render_id": render_id,
            "split": split,
            "season": season,
            "scene": normalize_scene_value(scene),
            "scene_label": scene_label(scene),
            "image_urls": {
                "cloudy": f"/asset/{render_id}/cloudy.png",
                "prediction": f"/asset/{render_id}/prediction.png" if has_prediction else None,
            },
            "active_layer": active_layer,
            "has_prediction": has_prediction,
            "width": int(layout["width"]),
            "height": int(layout["height"]),
            "patches": [
                {
                    "patch": str(item["patch"]),
                    "grid_row": int(item["grid_row"]),
                    "grid_col": int(item["grid_col"]),
                    "x": int((int(item["grid_col"]) - layout["min_col"]) * self.stride),
                    "y": int((int(item["grid_row"]) - layout["min_row"]) * self.stride),
                    "width": self.tile_size,
                    "height": self.tile_size,
                }
                for item in decoded_rows
            ],
            "stats": {
                "rows": int(rows),
                "blocks": int(blocks),
                "grid_rows": int(layout["grid_rows"]),
                "grid_cols": int(layout["grid_cols"]),
                "min_row": int(layout["min_row"]),
                "max_row": int(layout["max_row"]),
                "min_col": int(layout["min_col"]),
                "max_col": int(layout["max_col"]),
                "tile_size": self.tile_size,
                "stride": self.stride,
                "seconds": seconds,
                "rendered_at": utc_timestamp(),
                "source_mode": "streaming" if self.streaming else "local",
                "dataset_root": str(self.dataset_root) if self.dataset_root is not None else None,
                "patch_order": "top-left first; later rows and columns draw above earlier patches",
            },
            "model": self.predictor.model_meta,
        }


class InferenceJob:
    def __init__(
        self,
        *,
        job_id: str,
        renderer: SceneRenderer,
        item: dict[str, Any],
    ) -> None:
        self.job_id = job_id
        self.renderer = renderer
        self.item = item
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"infer-{job_id}", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def emit(self, payload: dict[str, Any]) -> None:
        event = dict(payload)
        event["job_id"] = self.job_id
        self.events.put(event)

    def _run(self) -> None:
        try:
            self.renderer.stream_infer_scene(
                split=str(self.item["split"]),
                season=str(self.item["season"]),
                scene=str(self.item["scene"]),
                send_event=self.emit,
            )
        except Exception as exc:
            self.emit({"type": "error", "error": str(exc)})
        finally:
            self.done.set()
            self.events.put({"type": "end", "job_id": self.job_id})


class InferenceJobManager:
    def __init__(self, *, renderer: SceneRenderer) -> None:
        self.renderer = renderer
        self.jobs: dict[str, InferenceJob] = {}
        self.lock = threading.Lock()

    def start_job(self, item: dict[str, Any]) -> InferenceJob:
        stable = (
            f"{item['split']}:{item['season']}:{normalize_scene_value(str(item['scene']))}:"
            f"{time.time_ns()}"
        )
        job_id = hashlib.sha1(stable.encode("utf-8")).hexdigest()[:16]
        job = InferenceJob(job_id=job_id, renderer=self.renderer, item=item)
        with self.lock:
            self.jobs[job_id] = job
        job.start()
        return job

    def get(self, job_id: str) -> InferenceJob:
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            raise DemoError(f"inference job not found: {job_id}")
        return job


WEB_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CLEAR-Net Scene Demo</title>
  <style>
    :root {
      --bg: #ffffff;
      --panel: #ffffff;
      --panel-soft: #f6f8fb;
      --text: #111827;
      --muted: #64748b;
      --line: #d8e0ea;
      --accent: #0ea5e9;
      --accent-2: #14b8a6;
      --danger: #b91c1c;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: var(--bg);
    }
    button { font: inherit; }
    .app {
      height: 100vh;
      overflow: hidden;
      display: block;
      background: var(--bg);
    }
    .progress-block {
      display: grid;
      gap: 6px;
      min-width: 0;
    }
    .progress-row {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 14px;
      font-size: 11px;
      color: var(--muted);
    }
    .progress-row strong {
      color: var(--text);
      font-size: 12px;
      font-weight: 640;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .track {
      height: 5px;
      border: 1px solid var(--line);
      border-radius: 99px;
      background: #eef3f8;
      overflow: hidden;
    }
    .bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      transition: width 360ms ease;
    }
    .actions {
      display: flex;
      align-items: center;
      justify-content: stretch;
      gap: 6px;
    }
    .button, .scene-select {
      height: 30px;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--text);
      font-size: 12px;
    }
    .button {
      background: #2563eb;
      border-color: #2563eb;
      color: white;
      padding: 0 11px;
      width: 100%;
      cursor: pointer;
      font-weight: 650;
    }
    .button:disabled {
      cursor: wait;
      color: #94a3b8;
      background: #e8edf3;
      border-color: #d8e0ea;
    }
    .nav-button {
      min-width: 0;
      width: 100%;
      padding: 0;
      background: var(--panel-soft);
      border-color: var(--line);
      color: var(--text);
    }
    .view-toggle {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }
    .view-button {
      height: 30px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #f8fafc;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      cursor: pointer;
    }
    .view-button.active {
      background: #111827;
      border-color: #111827;
      color: #ffffff;
    }
    .view-button:disabled {
      cursor: not-allowed;
      color: #94a3b8;
      background: #eef2f7;
      border-color: var(--line);
    }
    .scene-select {
      width: 100%;
      background: var(--panel-soft);
      padding: 0 8px;
      outline: none;
    }
    .scene-select:disabled {
      color: #94a3b8;
      background: #eef2f7;
    }
    .main {
      height: 100vh;
      min-height: 0;
      overflow: hidden;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 220px;
      gap: 6px;
      padding: 6px;
    }
    .viewport {
      min-height: 0;
      position: relative;
      display: grid;
      place-items: center;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
    }
    .scene-stage {
      position: relative;
      display: flex;
      max-width: 100%;
      max-height: 100%;
    }
    .sidebar {
      min-height: 0;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      gap: 12px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
    }
    .side-title {
      display: grid;
      gap: 4px;
      padding-bottom: 2px;
    }
    .side-title h1 {
      margin: 0;
      font-size: 15px;
      line-height: 1.1;
      font-weight: 760;
      letter-spacing: 0;
      color: var(--text);
    }
    .side-title p {
      margin: 0;
      min-height: 16px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.35;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .field {
      display: grid;
      gap: 5px;
    }
    .field label {
      font-size: 11px;
      font-weight: 650;
      color: var(--muted);
    }
    .scene-nav {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }
    .log-panel {
      min-height: 0;
      flex: 1;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 6px;
    }
    .log-panel h2 {
      margin: 0;
      color: var(--muted);
      font-size: 11px;
      font-weight: 650;
      line-height: 1.2;
    }
    .log-list {
      min-height: 0;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      justify-content: flex-end;
      gap: 4px;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #f8fafc;
    }
    .log-line {
      color: #334155;
      font-size: 11px;
      line-height: 1.25;
      word-break: break-word;
    }
    .log-line.muted {
      color: var(--muted);
    }
    .scene-image {
      max-width: 100%;
      max-height: calc(100vh - 14px);
      width: auto;
      height: auto;
      object-fit: contain;
      border: 1px solid #d8e0ea;
      background: #ffffff;
      image-rendering: auto;
      transition: opacity 180ms ease, filter 240ms ease;
    }
    .scene-image.loading {
      opacity: 1;
      filter: none;
    }
    .reveal-layer {
      position: absolute;
      inset: 0;
      overflow: hidden;
      pointer-events: none;
    }
    .blend-canvas {
      position: absolute;
      left: 0;
      top: 0;
      pointer-events: none;
    }
    .reveal-tile {
      position: absolute;
      opacity: 0;
      background-repeat: no-repeat;
      transition: opacity 90ms ease;
      will-change: opacity;
    }
    .reveal-tile.visible {
      opacity: 1;
    }
    .badge {
      position: absolute;
      left: 10px;
      top: 10px;
      display: flex;
      align-items: center;
      gap: 9px;
      padding: 6px 8px;
      border-radius: 6px;
      background: rgba(255,255,255,0.92);
      border: 1px solid var(--line);
      color: var(--text);
      box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
      font-size: 11px;
      pointer-events: none;
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 0 4px rgba(14,165,233,0.14);
    }
    .error { color: var(--danger); }
    @media (max-width: 760px) {
      .main { grid-template-columns: 1fr; }
      .sidebar { order: -1; }
      .actions { justify-content: stretch; }
      .button { flex: 1; }
      .scene-select { flex: 2; width: auto; }
      .scene-image { max-height: calc(100vh - 208px); }
    }
  </style>
</head>
<body>
  <div class="app">
    <main class="main">
      <section class="viewport">
        <div id="sceneStage" class="scene-stage">
          <img id="sceneImage" class="scene-image loading" alt="cloudy scene">
          <div id="revealLayer" class="reveal-layer"></div>
        </div>
        <div class="badge"><span class="dot"></span><span id="layerText">cloudy</span></div>
      </section>
      <aside class="sidebar">
        <div class="side-title">
          <h1>CLEAR-Net Demo</h1>
          <p id="sceneTitle">Loading scenes</p>
        </div>
        <div class="field">
          <label for="splitSelect">Split</label>
          <select id="splitSelect" class="scene-select" aria-label="Split"></select>
        </div>
        <div class="field">
          <label for="seasonSelect">Season</label>
          <select id="seasonSelect" class="scene-select" aria-label="Season"></select>
        </div>
        <div class="field">
          <label for="sceneSelect">Scene</label>
          <select id="sceneSelect" class="scene-select" aria-label="Scene"></select>
        </div>
        <div class="scene-nav">
          <button id="prevButton" class="button nav-button" type="button" aria-label="Previous scene">&lsaquo;</button>
          <button id="nextButton" class="button nav-button" type="button" aria-label="Next scene">&rsaquo;</button>
        </div>
        <div class="view-toggle" role="group" aria-label="Before and after view">
          <button id="beforeButton" class="view-button active" type="button" aria-pressed="true">Before</button>
          <button id="afterButton" class="view-button" type="button" aria-pressed="false" disabled>After</button>
        </div>
        <div class="actions">
          <button id="inferButton" class="button" type="button" disabled>Inference</button>
        </div>
        <div class="progress-block">
          <div class="progress-row">
            <strong>Patch Progress</strong>
            <span id="progressText">0 / 0</span>
          </div>
          <div class="track"><div id="progressBar" class="bar"></div></div>
        </div>
        <div class="log-panel">
          <h2>Log</h2>
          <div id="logList" class="log-list"></div>
        </div>
      </aside>
    </main>
  </div>
  <script>
    const state = {
      scenes: {},
      currentSplit: "validation",
      currentSeason: "spring",
      currentIndex: 0,
      completed: new Set(),
      patchDone: 0,
      patchTotal: 0,
      revealQueue: [],
      revealActive: false,
      finalRender: null,
      afterRender: null,
      afterPredictionUrl: null,
      afterReady: false,
      viewMode: "before",
      blend: null,
      busy: false,
    };

    const els = {
      image: document.getElementById("sceneImage"),
      stage: document.getElementById("sceneStage"),
      reveal: document.getElementById("revealLayer"),
      button: document.getElementById("inferButton"),
      sceneTitle: document.getElementById("sceneTitle"),
      progressText: document.getElementById("progressText"),
      progressBar: document.getElementById("progressBar"),
      layer: document.getElementById("layerText"),
      prev: document.getElementById("prevButton"),
      next: document.getElementById("nextButton"),
      select: document.getElementById("sceneSelect"),
      split: document.getElementById("splitSelect"),
      season: document.getElementById("seasonSelect"),
      before: document.getElementById("beforeButton"),
      after: document.getElementById("afterButton"),
      log: document.getElementById("logList"),
    };

    function setStatus(text, error=false) {
      if (error) {
        console.error(text);
        addLog(text, "muted");
      }
    }

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, ch => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;"
      }[ch]));
    }

    function activeQueue() {
      return ((state.scenes[state.currentSplit] || {})[state.currentSeason] || []);
    }

    function sceneKey(scene) {
      return `${scene.split}:${scene.season}:${scene.scene}`;
    }

    function sceneAt(index) {
      return activeQueue()[index] || null;
    }

    function updateProgress() {
      const queue = activeQueue();
      const total = queue.length;
      const patchTotal = Math.max(0, state.patchTotal);
      const patchDone = Math.min(Math.max(0, state.patchDone), patchTotal);
      const percent = patchTotal > 0 ? (patchDone / patchTotal) * 100 : 0;
      els.progressText.textContent = `${patchDone} / ${patchTotal}`;
      els.progressBar.style.width = `${percent}%`;
      const scene = sceneAt(state.currentIndex);
      if (scene) {
        els.sceneTitle.textContent = `${scene.scene_label} | ${scene.split} / ${scene.season}`;
        els.select.value = String(state.currentIndex);
        els.split.value = state.currentSplit;
        els.season.value = state.currentSeason;
      } else {
        els.sceneTitle.textContent = total > 0 ? "Select scene" : "No scenes";
        els.select.value = "";
      }
      els.button.disabled = state.busy || !scene;
      els.prev.disabled = state.busy || !scene || state.currentIndex <= 0;
      els.next.disabled = state.busy || !scene || state.currentIndex >= total - 1;
      els.select.disabled = state.busy || total <= 0;
      els.split.disabled = state.busy;
      els.season.disabled = state.busy;
      applyViewMode();
    }

    function setPatchProgress(done, total) {
      state.patchDone = done;
      state.patchTotal = total;
      updateProgress();
    }

    function resetRevealQueue() {
      state.revealQueue = [];
      state.revealActive = false;
      state.finalRender = null;
    }

    function clearLog() {
      els.log.innerHTML = "";
    }

    function addLog(text, tone="") {
      const line = document.createElement("div");
      line.className = tone ? `log-line ${tone}` : "log-line";
      line.textContent = text;
      els.log.appendChild(line);
      while (els.log.children.length > 42) {
        els.log.removeChild(els.log.firstChild);
      }
    }

    function fillSceneSelect() {
      els.select.innerHTML = "";
      if (state.currentIndex < 0) {
        const placeholder = document.createElement("option");
        placeholder.value = "";
        placeholder.textContent = "Select scene";
        placeholder.disabled = true;
        placeholder.selected = true;
        els.select.appendChild(placeholder);
      }
      for (const [index, scene] of activeQueue().entries()) {
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = scene.scene_label;
        els.select.appendChild(option);
      }
    }

    function fillSplitSelect() {
      els.split.innerHTML = "";
      for (const split of Object.keys(state.scenes)) {
        const option = document.createElement("option");
        option.value = split;
        option.textContent = split;
        els.split.appendChild(option);
      }
    }

    function fillSeasonSelect() {
      els.season.innerHTML = "";
      for (const season of Object.keys(state.scenes[state.currentSplit] || {})) {
        const option = document.createElement("option");
        option.value = season;
        option.textContent = season;
        els.season.appendChild(option);
      }
      const seasons = Object.keys(state.scenes[state.currentSplit] || {});
      if (!seasons.includes(state.currentSeason)) {
        state.currentSeason = seasons[0] || "spring";
      }
      els.season.value = state.currentSeason;
    }

    function imageUrl(url) {
      return `${url}?t=${Date.now()}`;
    }

    function sleep(ms) {
      return new Promise(resolve => setTimeout(resolve, ms));
    }

    function clearReveal() {
      els.reveal.innerHTML = "";
      state.blend = null;
    }

    function applyViewMode() {
      const afterVisible = state.viewMode === "after";
      const afterLabel = state.busy
        ? "after | inference"
        : (state.finalRender ? "after | prediction" : "after | partial");
      els.reveal.hidden = !afterVisible;
      els.before.classList.toggle("active", !afterVisible);
      els.after.classList.toggle("active", afterVisible);
      els.before.setAttribute("aria-pressed", String(!afterVisible));
      els.after.setAttribute("aria-pressed", String(afterVisible));
      els.after.disabled = !state.afterReady;
      els.layer.textContent = afterVisible
        ? afterLabel
        : "before | cloudy";
    }

    function setViewMode(mode) {
      if (mode === "after" && !state.afterReady) return;
      state.viewMode = mode === "after" ? "after" : "before";
      applyViewMode();
    }

    function resetSceneView() {
      resetRevealQueue();
      clearReveal();
      state.afterRender = null;
      state.afterPredictionUrl = null;
      state.afterReady = false;
      state.viewMode = "before";
      applyViewMode();
    }

    function markAfterReady(render=null) {
      if (render) state.afterRender = render;
      state.afterReady = true;
      applyViewMode();
    }

    function preloadImage(src) {
      return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = reject;
        image.src = src;
      });
    }

    function waitForDisplayedImage() {
      if (els.image.complete && els.image.naturalWidth > 0) return Promise.resolve();
      return new Promise((resolve, reject) => {
        els.image.onload = () => resolve();
        els.image.onerror = reject;
      });
    }

    function prepareRevealLayer(renderWidth, renderHeight) {
      const imageRect = els.image.getBoundingClientRect();
      els.reveal.style.width = `${imageRect.width}px`;
      els.reveal.style.height = `${imageRect.height}px`;
      els.reveal.style.background = "transparent";
      return {
        width: imageRect.width,
        height: imageRect.height,
        sx: imageRect.width / renderWidth,
        sy: imageRect.height / renderHeight,
      };
    }

    function hannWeights(width, height) {
      const edge = 0.15;
      const xWeights = new Float32Array(width);
      const yWeights = new Float32Array(height);
      for (let x = 0; x < width; x++) {
        const phase = width <= 1 ? Math.PI / 2 : Math.PI * x / (width - 1);
        xWeights[x] = edge + (1 - edge) * Math.sin(phase);
      }
      for (let y = 0; y < height; y++) {
        const phase = height <= 1 ? Math.PI / 2 : Math.PI * y / (height - 1);
        yWeights[y] = edge + (1 - edge) * Math.sin(phase);
      }
      const weights = new Float32Array(width * height);
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          weights[y * width + x] = yWeights[y] * xWeights[x];
        }
      }
      return weights;
    }

    function ensureBlend(renderWidth, renderHeight) {
      const scale = prepareRevealLayer(renderWidth, renderHeight);
      const current = state.blend;
      if (current && current.width === renderWidth && current.height === renderHeight) {
        current.canvas.style.width = `${scale.width}px`;
        current.canvas.style.height = `${scale.height}px`;
        return current;
      }

      clearReveal();
      const canvas = document.createElement("canvas");
      canvas.className = "blend-canvas";
      canvas.width = renderWidth;
      canvas.height = renderHeight;
      canvas.style.width = `${scale.width}px`;
      canvas.style.height = `${scale.height}px`;
      els.reveal.appendChild(canvas);

      state.blend = {
        canvas,
        ctx: canvas.getContext("2d"),
        width: renderWidth,
        height: renderHeight,
        accum: new Float32Array(renderWidth * renderHeight * 3),
        weights: new Float32Array(renderWidth * renderHeight),
        weightCache: new Map(),
      };
      return state.blend;
    }

    function tileWeights(width, height, blend) {
      const key = `${width}x${height}`;
      if (!blend.weightCache.has(key)) {
        blend.weightCache.set(key, hannWeights(width, height));
      }
      return blend.weightCache.get(key);
    }

    async function showPredictionTile(event) {
      const blend = ensureBlend(event.render_width, event.render_height);
      const image = await preloadImage(imageUrl(event.tile_url));
      const tileCanvas = document.createElement("canvas");
      tileCanvas.width = event.width;
      tileCanvas.height = event.height;
      const tileCtx = tileCanvas.getContext("2d");
      tileCtx.drawImage(image, 0, 0, event.width, event.height);
      const tileData = tileCtx.getImageData(0, 0, event.width, event.height).data;
      const weights = tileWeights(event.width, event.height, blend);
      const output = blend.ctx.createImageData(event.width, event.height);

      for (let y = 0; y < event.height; y++) {
        const sceneY = event.y + y;
        if (sceneY < 0 || sceneY >= blend.height) continue;
        for (let x = 0; x < event.width; x++) {
          const sceneX = event.x + x;
          if (sceneX < 0 || sceneX >= blend.width) continue;
          const tileIndex = y * event.width + x;
          const tileBase = tileIndex * 4;
          const sceneIndex = sceneY * blend.width + sceneX;
          const sceneBase = sceneIndex * 3;
          const weight = weights[tileIndex];

          blend.accum[sceneBase] += tileData[tileBase] * weight;
          blend.accum[sceneBase + 1] += tileData[tileBase + 1] * weight;
          blend.accum[sceneBase + 2] += tileData[tileBase + 2] * weight;
          blend.weights[sceneIndex] += weight;

          const outBase = tileBase;
          const totalWeight = blend.weights[sceneIndex] || 1;
          output.data[outBase] = Math.round(blend.accum[sceneBase] / totalWeight);
          output.data[outBase + 1] = Math.round(blend.accum[sceneBase + 1] / totalWeight);
          output.data[outBase + 2] = Math.round(blend.accum[sceneBase + 2] / totalWeight);
          output.data[outBase + 3] = 255;
        }
      }
      blend.ctx.putImageData(output, event.x, event.y);
    }

    async function enqueuePredictionTile(event) {
      state.revealQueue.push(event);
      if (state.revealActive) return;

      state.revealActive = true;
      while (state.revealQueue.length) {
        const nextEvent = state.revealQueue.shift();
        await showPredictionTile(nextEvent);
        setPatchProgress(Number(nextEvent.index || 0), Number(nextEvent.total || 0));
        addLog(`${nextEvent.patch} complete | row ${nextEvent.grid_row}, col ${nextEvent.grid_col}`);
        await sleep(18);
      }
      state.revealActive = false;
    }

    async function waitForRevealQueue() {
      while (state.revealActive || state.revealQueue.length) {
        await sleep(20);
      }
    }

    async function revealPrediction(render) {
      const predictionUrl = imageUrl(render.image_urls.prediction);
      await waitForDisplayedImage();
      await preloadImage(predictionUrl);
      state.finalRender = render;
      markAfterReady(render);
      state.afterPredictionUrl = predictionUrl;
      clearReveal();

      const patches = [...(render.patches || [])].sort((a, b) => (
        (Number(a.grid_row) - Number(b.grid_row)) ||
        (Number(a.grid_col) - Number(b.grid_col)) ||
        String(a.patch).localeCompare(String(b.patch))
      ));
      if (!patches.length) {
        const scale = prepareRevealLayer(render.width, render.height);
        const prediction = document.createElement("img");
        prediction.className = "blend-canvas";
        prediction.src = predictionUrl;
        prediction.style.width = `${scale.width}px`;
        prediction.style.height = `${scale.height}px`;
        els.reveal.appendChild(prediction);
        return;
      }

      const imageRect = els.image.getBoundingClientRect();
      const scale = prepareRevealLayer(render.width, render.height);
      const sx = scale.sx;
      const sy = scale.sy;
      const tileDelay = Math.max(4, Math.min(24, Math.floor(2800 / patches.length)));
      setPatchProgress(0, patches.length);
      addLog(`${render.scene_label} prediction tiles ready`, "muted");

      const fragment = document.createDocumentFragment();
      const tiles = patches.map(patch => {
        const tile = document.createElement("div");
        tile.className = "reveal-tile";
        tile.style.left = `${patch.x * sx}px`;
        tile.style.top = `${patch.y * sy}px`;
        tile.style.width = `${patch.width * sx}px`;
        tile.style.height = `${patch.height * sy}px`;
        tile.style.backgroundImage = `url("${predictionUrl}")`;
        tile.style.backgroundSize = `${imageRect.width}px ${imageRect.height}px`;
        tile.style.backgroundPosition = `${-patch.x * sx}px ${-patch.y * sy}px`;
        fragment.appendChild(tile);
        return tile;
      });
      els.reveal.appendChild(fragment);

      for (const [index, tile] of tiles.entries()) {
        const patch = patches[index];
        tile.classList.add("visible");
        setPatchProgress(index + 1, patches.length);
        addLog(`${patch.patch} complete | row ${patch.grid_row}, col ${patch.grid_col}`);
        await sleep(tileDelay);
      }
      await sleep(160);
      addLog(`${render.scene_label} complete`, "muted");
    }

    async function handleInferEvent(event) {
      if (event.type === "status") {
        addLog(event.message || "working", "muted");
        return null;
      }
      if (event.type === "start") {
        const render = event.render;
        await waitForDisplayedImage();
        clearReveal();
        resetRevealQueue();
        markAfterReady(render);
        setPatchProgress(0, Number(event.total || 0));
        addLog(`${render.scene_label} started: ${event.total} patches`, "muted");
        return null;
      }
      if (event.type === "patch") {
        markAfterReady();
        await enqueuePredictionTile(event);
        return null;
      }
      if (event.type === "cached") {
        await revealPrediction(event.render);
        return event.render;
      }
      if (event.type === "complete") {
        const render = event.render;
        const predictionUrl = imageUrl(render.image_urls.prediction);
        state.finalRender = render;
        markAfterReady(render);
        await waitForRevealQueue();
        await preloadImage(predictionUrl);
        state.afterPredictionUrl = predictionUrl;
        setPatchProgress((render.patches || []).length, (render.patches || []).length);
        addLog(`${render.scene_label} complete`, "muted");
        return render;
      }
      if (event.type === "error") {
        throw new Error(event.error || "inference failed");
      }
      return null;
    }

    async function streamInference(scene) {
      const start = await postJson("/api/infer-start", scene);
      const params = new URLSearchParams({job_id: start.job_id});
      return new Promise((resolve, reject) => {
        const source = new EventSource(`/api/infer-events?${params.toString()}`);
        let settled = false;

        source.onmessage = async message => {
          try {
            const event = JSON.parse(message.data);
            if (event.type === "heartbeat") return;
            const render = await handleInferEvent(event);
            if (event.type === "cached" || event.type === "complete") {
              settled = true;
              source.close();
              resolve(render || event.render);
            }
          } catch (error) {
            settled = true;
            source.close();
            reject(error);
          }
        };

        source.onerror = () => {
          source.close();
          if (!settled) {
            reject(new Error("inference event stream disconnected"));
          }
        };
      });
    }

    async function postJson(path, payload) {
      const response = await fetch(path, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || response.statusText);
      return data;
    }

    async function loadScene(index) {
      const scene = sceneAt(index);
      if (!scene) {
        setStatus("<strong>Done</strong> all scenes processed");
        els.layer.textContent = "complete";
        updateProgress();
        return;
      }
      state.busy = true;
      updateProgress();
      resetSceneView();
      setPatchProgress(0, Number(scene.rows || 0));
      clearLog();
      addLog(`${scene.scene_label} cloudy scene loading`, "muted");
      setStatus(`<strong>Loading cloudy scene</strong> ${escapeHtml(scene.scene_label)}`);
      try {
        const render = await postJson("/api/scene", scene);
        els.image.src = imageUrl(render.image_urls.cloudy);
        els.image.alt = `${render.scene_label} cloudy`;
        setPatchProgress(0, (render.patches || []).length || Number(render.stats?.rows || scene.rows || 0));
        addLog(`${render.scene_label} ready: ${state.patchTotal} patches`, "muted");
        setStatus(`<strong>Ready</strong> ${escapeHtml(render.scene_label)} cloudy scene`);
      } catch (error) {
        setStatus(String(error.message || error), true);
      } finally {
        state.busy = false;
        updateProgress();
      }
    }

    async function inferCurrent() {
      const index = state.currentIndex;
      const scene = sceneAt(index);
      if (!scene || state.busy) return;
      state.busy = true;
      state.afterReady = true;
      setViewMode("after");
      updateProgress();
      setPatchProgress(0, Number(scene.rows || state.patchTotal || 0));
      clearLog();
      addLog(`${scene.scene_label} inference requested`, "muted");
      setStatus(`<strong>Running inference</strong> ${escapeHtml(scene.scene_label)}`);
      try {
        const render = await streamInference(scene);
        if (!render) throw new Error("inference stream ended before completion");
        els.image.alt = `${render.scene_label} cloudy base with prediction overlay`;
        state.completed.add(sceneKey(scene));
        updateProgress();
        setStatus(`<strong>Inference complete</strong> ${escapeHtml(render.scene_label)}`);
      } catch (error) {
        setStatus(String(error.message || error), true);
      } finally {
        state.busy = false;
        updateProgress();
      }
    }

    function changeScene(index) {
      const total = activeQueue().length;
      if (state.busy || !Number.isFinite(index) || total <= 0) return;
      const target = Math.max(0, Math.min(total - 1, index));
      if (state.busy || target === state.currentIndex) return;
      state.currentIndex = target;
      loadScene(target);
    }

    function changeSplit(split) {
      if (state.busy || split === state.currentSplit) return;
      state.currentSplit = split;
      fillSeasonSelect();
      state.currentIndex = -1;
      fillSceneSelect();
      setPatchProgress(0, 0);
      clearLog();
      addLog(`${state.currentSplit} / ${state.currentSeason} selected`, "muted");
      addLog("Choose a scene to load", "muted");
    }

    function changeSeason(season) {
      if (state.busy || season === state.currentSeason) return;
      state.currentSeason = season;
      state.currentIndex = -1;
      fillSceneSelect();
      setPatchProgress(0, 0);
      clearLog();
      addLog(`${state.currentSplit} / ${state.currentSeason} selected`, "muted");
      addLog("Choose a scene to load", "muted");
    }

    async function init() {
      try {
        const payload = await fetch("/api/state").then(response => {
          if (!response.ok) throw new Error(response.statusText);
          return response.json();
        });
        state.scenes = payload.scenes || payload.scenes_by_split || {};
        state.currentSplit = payload.default_split || Object.keys(state.scenes)[0] || "validation";
        state.currentSeason = payload.default_season || payload.season || Object.keys(state.scenes[state.currentSplit] || {})[0] || "spring";
        state.currentIndex = 0;
        state.completed = new Set();
        fillSplitSelect();
        fillSeasonSelect();
        fillSceneSelect();
        setStatus(`<strong>Loaded</strong> ${activeQueue().length} scenes`);
        updateProgress();
        await loadScene(0);
      } catch (error) {
        setStatus(String(error.message || error), true);
      }
    }

    els.button.addEventListener("click", inferCurrent);
    els.before.addEventListener("click", () => setViewMode("before"));
    els.after.addEventListener("click", () => setViewMode("after"));
    els.prev.addEventListener("click", () => changeScene(state.currentIndex - 1));
    els.next.addEventListener("click", () => changeScene(state.currentIndex + 1));
    els.select.addEventListener("change", () => {
      if (els.select.value === "") return;
      changeScene(Number(els.select.value));
    });
    els.split.addEventListener("change", () => changeSplit(els.split.value));
    els.season.addEventListener("change", () => changeSeason(els.season.value));
    document.addEventListener("keydown", event => {
      if (event.key === " " || event.key === "Enter") {
        event.preventDefault();
        inferCurrent();
      }
    });

    init();
  </script>
</body>
</html>
"""


class DemoServer:
    def __init__(
        self,
        *,
        queue: list[dict[str, Any]],
        scenes_by_split: dict[str, Any],
        default_split: str,
        season: str,
        renderer: SceneRenderer,
        jobs: InferenceJobManager,
    ) -> None:
        self.queue = queue
        self.scenes_by_split = scenes_by_split
        self.default_split = default_split
        self.season = season
        self.renderer = renderer
        self.jobs = jobs

    def queue_item(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self.queue):
            raise DemoError(f"scene index out of range: {index}")
        return self.queue[index]

    def payload_item(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "split" not in payload or "scene" not in payload:
            return self.queue_item(int(payload.get("index", 0)))
        split = str(payload["split"])
        scene = normalize_scene_value(str(payload["scene"]))
        season = str(payload.get("season") or self.season)
        if split not in SPLITS:
            raise DemoError(f"unsupported split: {split!r}")
        return {
            "split": split,
            "season": season,
            "scene": scene,
            "scene_label": scene_label(scene),
        }

    def handler(self):
        server_state = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, _format: str, *args: Any) -> None:
                return

            def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def send_file(self, path: Path) -> None:
                data = path.read_bytes()
                content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def send_sse_event(self, payload: dict[str, Any]) -> None:
                data = "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
                self.wfile.write(data.encode("utf-8"))
                self.wfile.flush()

            def read_body_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0:
                    return {}
                return json.loads(self.rfile.read(length).decode("utf-8"))

            def do_GET(self) -> None:
                parsed = urllib.parse.urlparse(self.path)
                try:
                    if parsed.path == "/":
                        data = WEB_HTML.encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.send_header("Content-Length", str(len(data)))
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        self.wfile.write(data)
                        return
                    if parsed.path == "/favicon.ico":
                        self.send_response(204)
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        return
                    if parsed.path == "/api/state":
                        self.send_json(
                            {
                                "queue": server_state.queue,
                                "scenes": server_state.scenes_by_split,
                                "scenes_by_split": server_state.scenes_by_split,
                                "default_split": server_state.default_split,
                                "default_season": server_state.season,
                                "season": server_state.season,
                                "seasons": list(SEASONS),
                                "total": len(server_state.queue),
                                "generated_at": utc_timestamp(),
                            }
                        )
                        return
                    if parsed.path == "/api/infer-stream":
                        params = urllib.parse.parse_qs(parsed.query)
                        payload = {key: str((value or [""])[0]) for key, value in params.items()}
                        item = server_state.payload_item(payload)
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("Connection", "keep-alive")
                        self.send_header("X-Accel-Buffering", "no")
                        self.end_headers()
                        try:
                            server_state.renderer.stream_infer_scene(
                                split=str(item["split"]),
                                season=str(item["season"]),
                                scene=str(item["scene"]),
                                send_event=self.send_sse_event,
                            )
                        except BrokenPipeError:
                            return
                        except Exception as exc:
                            self.send_sse_event({"type": "error", "error": str(exc)})
                        return
                    if parsed.path == "/api/infer-events":
                        params = urllib.parse.parse_qs(parsed.query)
                        job_id = str((params.get("job_id") or [""])[0])
                        job = server_state.jobs.get(job_id)
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("Connection", "keep-alive")
                        self.send_header("X-Accel-Buffering", "no")
                        self.end_headers()
                        try:
                            while True:
                                try:
                                    event = job.events.get(timeout=15)
                                except queue.Empty:
                                    self.send_sse_event({"type": "heartbeat", "job_id": job_id})
                                    continue
                                if event.get("type") == "end":
                                    break
                                self.send_sse_event(event)
                        except BrokenPipeError:
                            return
                        return
                    if parsed.path.startswith("/asset/"):
                        parts = parsed.path.split("/")
                        if len(parts) not in {4, 5}:
                            self.send_error(404, "asset not found")
                            return
                        render_id = parts[2]
                        if len(parts) == 5 and parts[3] == "tiles":
                            asset_name = f"tiles/{parts[4]}"
                            allowed = asset_name.endswith(".png")
                        else:
                            asset_name = parts[3]
                            allowed = asset_name in {"cloudy.png", "prediction.png"}
                        if not allowed:
                            self.send_error(404, "asset not found")
                            return
                        asset_path = server_state.renderer.render_dir(render_id) / asset_name
                        if not asset_path.is_file():
                            self.send_error(404, "asset not found")
                            return
                        self.send_file(asset_path)
                        return
                    self.send_error(404)
                except Exception as exc:
                    self.send_json({"error": str(exc)}, status=400)

            def do_POST(self) -> None:
                parsed = urllib.parse.urlparse(self.path)
                try:
                    payload = self.read_body_json()
                    if parsed.path == "/api/infer-start":
                        item = server_state.payload_item(payload)
                        job = server_state.jobs.start_job(item)
                        self.send_json({"job_id": job.job_id})
                        return
                    if parsed.path in {"/api/scene", "/api/infer"}:
                        item = server_state.payload_item(payload)
                        render = server_state.renderer.render_scene(
                            split=str(item["split"]),
                            season=str(item["season"]),
                            scene=str(item["scene"]),
                            infer=parsed.path == "/api/infer",
                        )
                        self.send_json(render)
                        return
                    self.send_error(404)
                except Exception as exc:
                    self.send_json({"error": str(exc)}, status=400)

        return Handler


def parse_scene_list(value: str | None) -> tuple[str, ...] | None:
    if value is None or not value.strip():
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local CLEAR-Net scene inference demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8797)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=SPLITS, default="validation")
    parser.add_argument("--season", choices=SEASONS, default="spring")
    parser.add_argument("--scenes", default=None, help="Comma-separated scene ids. Defaults to the first scenes.")
    parser.add_argument("--max-scenes", type=int, default=12)
    parser.add_argument("--tile-size", type=int, default=SCENE_TILE_SIZE)
    parser.add_argument("--infer-batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--mixed-precision", choices=("off", "bf16", "fp16"), default="bf16")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--no-open-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tile_size <= 0:
        raise DemoError("--tile-size must be greater than zero")
    if args.infer_batch_size <= 0:
        raise DemoError("--infer-batch-size must be greater than zero")
    if not args.checkpoint.is_file():
        raise DemoError(f"checkpoint not found: {args.checkpoint}")

    dataset_root = Path(args.dataset_root) if args.dataset_root is not None else None
    local_ready = bool(
        dataset_root is not None
        and (dataset_root / "manifest.json").is_file()
        and (dataset_root / "catalogs").is_dir()
    )
    streaming = bool(args.streaming or not local_ready)
    if streaming:
        dataset_root = None
    if not streaming and not local_ready:
        raise DemoError(f"dataset root is unavailable: {args.dataset_root}")

    device = resolve_device(args.device)
    checkpoint_hash = checkpoint_fingerprint(args.checkpoint)
    predictor = ClearNetPredictor(
        checkpoint_path=args.checkpoint,
        device=device,
        mixed_precision=args.mixed_precision,
        batch_size=args.infer_batch_size,
    )
    catalog = SceneCatalog(streaming=streaming, dataset_root=dataset_root)
    queue = catalog.build_queue(
        split=args.split,
        season=args.season,
        scenes=parse_scene_list(args.scenes),
        max_scenes=args.max_scenes,
    )
    scenes_by_split = catalog.build_scene_options(
        max_scenes=args.max_scenes,
    )
    renderer = SceneRenderer(
        catalog=catalog,
        predictor=predictor,
        cache_dir=args.cache_dir,
        tile_size=args.tile_size,
        streaming=streaming,
        dataset_root=dataset_root,
        checkpoint_hash=checkpoint_hash,
    )
    jobs = InferenceJobManager(renderer=renderer)
    demo = DemoServer(
        queue=queue,
        scenes_by_split=scenes_by_split,
        default_split=args.split,
        season=args.season,
        renderer=renderer,
        jobs=jobs,
    )

    server = None
    selected_port = int(args.port)
    for candidate in range(int(args.port), int(args.port) + 30):
        try:
            server = http.server.ThreadingHTTPServer((args.host, candidate), demo.handler())
            selected_port = candidate
            break
        except OSError:
            continue
    if server is None:
        raise DemoError(f"no available port from {args.port} to {args.port + 29}")

    url = f"http://{args.host}:{selected_port}/"
    print(f"CLEAR-Net scene demo: {url}", flush=True)
    print(f"checkpoint: {args.checkpoint}", flush=True)
    print("model: CLEAR_Net(return_decomposition=True)", flush=True)
    print("data source: " + ("streaming" if streaming else f"local ({dataset_root})"), flush=True)
    print(f"queue: {args.split}/{args.season}, scenes={len(queue)}", flush=True)
    print(f"device: {device}, mixed_precision={args.mixed_precision}", flush=True)
    if not args.no_open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
