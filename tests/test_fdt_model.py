from __future__ import annotations

import torch

from modules.model.fdt import FDT


def test_fdt_imports_and_runs_forward() -> None:
    model = FDT(dim=16, num_layers=1, num_heads=4).eval()
    sar = torch.randn(1, 2, 16, 16)
    cloudy = torch.randn(1, 13, 16, 16)

    with torch.no_grad():
        outputs = model(sar, cloudy)

    assert len(outputs) == 5
    for output in outputs:
        assert output.shape[0] == sar.shape[0]
        assert output.ndim == 4
        assert output.dtype == cloudy.dtype
        assert bool(torch.isfinite(output).all().item())
