from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import torch.nn as nn
from cr_train import is_primary

from modules.model.fdt import FDT
from modules.model.fdt.fdt import CommonGate


FDTGateLevel = Literal["lowres", "midres", "highres"]
FDTGateBranch = Literal["sar", "cloudy"]
FDTGateStage = Literal["train", "val", "test"]
GateStatValue = float | int | str

_LEVEL_GATE_ATTRS: Mapping[
    FDTGateLevel,
    tuple[tuple[FDTGateBranch, str], tuple[FDTGateBranch, str]],
] = {
    "lowres": (
        ("sar", "sar_low_common_gate"),
        ("cloudy", "cld_low_common_gate"),
    ),
    "midres": (
        ("sar", "sar_mid_common_gate"),
        ("cloudy", "cld_mid_common_gate"),
    ),
    "highres": (
        ("sar", "sar_high_common_gate"),
        ("cloudy", "cld_high_common_gate"),
    ),
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


def _iter_level_gates(
    fdt: FDT,
    *,
    levels: Iterable[FDTGateLevel],
) -> Iterable[tuple[FDTGateLevel, FDTGateBranch, CommonGate]]:
    for level in levels:
        if level not in _LEVEL_GATE_ATTRS:
            raise ValueError(f"unsupported FDT gate level: {level}")
        for branch, attr_name in _LEVEL_GATE_ATTRS[level]:
            gate = getattr(fdt, attr_name)
            if not isinstance(gate, CommonGate):
                raise TypeError(f"expected CommonGate at FDT.{attr_name}")
            yield level, branch, gate


def collect_fdt_gate_stats(
    model: nn.Module,
    *,
    levels: Iterable[FDTGateLevel] = ("highres",),
) -> list[dict[str, GateStatValue]]:
    fdt = find_fdt_module(model)
    if fdt is None:
        return []

    records: list[dict[str, GateStatValue]] = []
    for level, branch, gate in _iter_level_gates(fdt, levels=tuple(levels)):
        if gate.last_gate_stats is None:
            continue
        for gate_name, stats in gate.last_gate_stats.items():
            if gate_name not in {"channel", "spatial"}:
                continue
            records.append(
                {
                    "level": level,
                    "branch": branch,
                    "gate": gate_name,
                    **stats,
                }
            )
    return records


def append_fdt_gate_diagnostics(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
    *,
    epoch: int,
    global_step: int,
    stage: FDTGateStage,
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
