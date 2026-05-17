from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import torch.nn as nn
from cr_train import is_primary

from modules.model.fdt import FDT
from modules.model.fdt.fdt import (
    BidirectionalDecompositionBlock,
    DecompositionBlock,
)


FDTDiagnosticLevel = Literal["lowres", "midres", "highres"]
FDTDiagnosticBranch = Literal["sar", "cloudy"]
FDTDiagnosticStage = Literal["train", "val", "test"]
DiagnosticValue = float | int | str

_LEVEL_DECOMP_ATTRS: Mapping[FDTDiagnosticLevel, str] = {
    "lowres": "low_decomp",
    "midres": "mid_decomp",
    "highres": "high_decomp",
}
_BRANCH_DECOMP_ATTRS: Mapping[FDTDiagnosticBranch, str] = {
    "sar": "sar",
    "cloudy": "cloudy",
}


def _unwrap_model(model: nn.Module) -> nn.Module:
    while hasattr(model, "module") and isinstance(model.module, nn.Module):
        model = model.module
    return model


def find_fdt_module(model: nn.Module) -> FDT | None:
    model = _unwrap_model(model)
    if isinstance(model, FDT):
        return model

    fdt = getattr(model, "fdt", None)
    if isinstance(fdt, FDT):
        return fdt

    for module in model.modules():
        if isinstance(module, FDT):
            return module
    return None


def _iter_level_decomps(
    fdt: FDT,
    *,
    levels: Iterable[FDTDiagnosticLevel],
) -> Iterable[tuple[FDTDiagnosticLevel, FDTDiagnosticBranch, DecompositionBlock]]:
    for level in levels:
        if level not in _LEVEL_DECOMP_ATTRS:
            raise ValueError(f"unsupported FDT diagnostic level: {level}")
        pair_attr_name = _LEVEL_DECOMP_ATTRS[level]
        pair = getattr(fdt, pair_attr_name)
        if not isinstance(pair, BidirectionalDecompositionBlock):
            raise TypeError(
                f"expected BidirectionalDecompositionBlock at FDT.{pair_attr_name}"
            )
        for branch, branch_attr_name in _BRANCH_DECOMP_ATTRS.items():
            decomp = getattr(pair, branch_attr_name)
            if not isinstance(decomp, DecompositionBlock):
                raise TypeError(
                    f"expected DecompositionBlock at FDT.{pair_attr_name}.{branch_attr_name}"
                )
            yield level, branch, decomp


def collect_fdt_relevance_stats(
    model: nn.Module,
    *,
    levels: Iterable[FDTDiagnosticLevel] = ("highres",),
) -> list[dict[str, DiagnosticValue]]:
    fdt = find_fdt_module(model)
    if fdt is None:
        return []

    records: list[dict[str, DiagnosticValue]] = []
    for level, branch, decomp in _iter_level_decomps(fdt, levels=tuple(levels)):
        if decomp.last_relevance_stats is None:
            continue
        for axis, stats in decomp.last_relevance_stats.items():
            if axis not in {"channel", "spatial"}:
                continue
            records.append(
                {
                    "level": level,
                    "branch": branch,
                    "kind": "relevance",
                    "axis": axis,
                    **stats,
                }
            )
    return records


def collect_fdt_gate_stats(
    model: nn.Module,
    *,
    levels: Iterable[FDTDiagnosticLevel] = ("highres",),
) -> list[dict[str, DiagnosticValue]]:
    return collect_fdt_relevance_stats(model, levels=levels)


def append_fdt_gate_diagnostics(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
    *,
    epoch: int,
    global_step: int,
    stage: FDTDiagnosticStage,
    primary_only: bool = True,
) -> None:
    if primary_only and not is_primary():
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            payload = {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "stage": stage,
                **record,
            }
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
