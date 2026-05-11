from __future__ import annotations

import torch
import torch.nn as nn

from modules.model.fdt import FDT, FDT_CRNet_Direct, FDT_CRNet_Side, ShuffleUp


def test_fdt_imports_and_runs_forward() -> None:
    model = FDT(num_layers=1, num_heads=4).eval()
    sar = torch.randn(1, 2, 16, 16)
    cloudy = torch.randn(1, 13, 16, 16)

    with torch.no_grad():
        outputs = model(sar, cloudy)

    assert len(outputs) == 5
    assert outputs[0].shape == (1, 256, 16, 16)
    for output in outputs[1:]:
        assert output.shape == (1, 64, 16, 16)
    for output in outputs:
        assert output.dtype == cloudy.dtype
        assert bool(torch.isfinite(output).all().item())


def test_fdt_uses_separate_shuffle_up_modules() -> None:
    model = FDT(num_layers=1, num_heads=4).eval()

    assert isinstance(model.feat_up, ShuffleUp)
    assert isinstance(model.gate_up, ShuffleUp)
    assert model.feat_up is not model.gate_up


def test_fdt_encoders_share_architecture_not_weights() -> None:
    model = FDT(num_layers=1, num_heads=4).eval()

    assert type(model.sar_encoder) is type(model.cld_encoder)
    assert model.sar_encoder is not model.cld_encoder

    sar_proj = model.sar_encoder.proj
    cld_proj = model.cld_encoder.proj

    assert isinstance(sar_proj[0], nn.Conv2d)
    assert sar_proj[0].kernel_size == (1, 1)
    assert sar_proj[0].in_channels == 2
    assert sar_proj[0].out_channels == model.dim

    assert isinstance(cld_proj[0], nn.Conv2d)
    assert cld_proj[0].kernel_size == (1, 1)
    assert cld_proj[0].in_channels == 13
    assert cld_proj[0].out_channels == model.dim

    assert isinstance(sar_proj[2], nn.Conv2d)
    assert sar_proj[2].kernel_size == (3, 3)
    assert sar_proj[2].stride == (1, 1)
    assert isinstance(sar_proj[4], nn.Conv2d)
    assert sar_proj[4].kernel_size == (3, 3)
    assert sar_proj[4].stride == (2, 2)


def test_fdt_uses_directional_common_gates() -> None:
    model = FDT(num_layers=1, num_heads=4).eval()

    assert model.sar_common_gate is not model.cld_common_gate
    for gate in (model.sar_common_gate, model.cld_common_gate):
        assert isinstance(gate.gate_head[0], nn.Conv2d)
        assert gate.gate_head[0].in_channels == model.dim * 2
        assert gate.gate_head[0].out_channels == model.dim
        assert isinstance(gate.gate_head[-1], nn.Conv2d)
        assert gate.gate_head[-1].in_channels == model.dim
        assert gate.gate_head[-1].out_channels == model.dim
        assert torch.count_nonzero(gate.gate_head[-1].weight).item() == 0
        assert torch.count_nonzero(gate.gate_head[-1].bias).item() == 0


def test_shuffle_up_is_initially_pixel_shuffle() -> None:
    up = ShuffleUp(256).eval()
    feature = torch.randn(2, 256, 8, 8)

    with torch.no_grad():
        actual = up(feature)
        expected = torch.nn.functional.pixel_shuffle(feature, 2)

    assert actual.shape == (2, 64, 16, 16)
    assert torch.allclose(actual, expected)


def test_fdt_crnet_direct_imports_and_runs_forward() -> None:
    model = FDT_CRNet_Direct(fdt_layers=1, cr_layers=1).eval()
    sar = torch.randn(1, 2, 16, 16)
    cloudy = torch.randn(1, 13, 16, 16)

    with torch.no_grad():
        prediction = model(sar, cloudy)

    assert prediction.shape == cloudy.shape
    assert prediction.dtype == cloudy.dtype


def test_fdt_crnet_side_imports_and_runs_forward() -> None:
    model = FDT_CRNet_Side(fdt_layers=1, cr_layers=1).eval()
    sar = torch.randn(1, 2, 16, 16)
    cloudy = torch.randn(1, 13, 16, 16)

    with torch.no_grad():
        prediction = model(sar, cloudy)

    assert prediction.shape == cloudy.shape
    assert prediction.dtype == cloudy.dtype
