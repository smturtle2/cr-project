from __future__ import annotations

import torch
import torch.nn as nn

from modules.model.fdt import FDT, FDT_CRNet_Direct, FDT_CRNet_Side, ResizeConvUp
from modules.model.module.attention import MultiHeadAttention


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


def test_fdt_uses_single_shared_resize_conv_up_module() -> None:
    model = FDT(num_layers=1, num_heads=4).eval()

    assert isinstance(model.up, ResizeConvUp)
    assert not hasattr(model, "feat_up")
    assert not hasattr(model, "gate_up")


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
        assert isinstance(gate.cross_attn, MultiHeadAttention)
        assert isinstance(gate.gate_head[0], nn.Conv2d)
        assert gate.gate_head[0].in_channels == model.dim
        assert gate.gate_head[0].out_channels == model.dim
        assert gate.gate_head[0].kernel_size == (3, 3)
        assert isinstance(gate.gate_head[-1], nn.Conv2d)
        assert gate.gate_head[-1].in_channels == model.dim
        assert gate.gate_head[-1].out_channels == model.dim
        assert gate.gate_head[-1].kernel_size == (1, 1)
        assert torch.count_nonzero(gate.gate_head[-1].weight).item() == 0
        assert torch.count_nonzero(gate.gate_head[-1].bias).item() == 0


def test_fdt_defines_comp_as_high_res_residual() -> None:
    model = FDT(num_layers=1, num_heads=4).eval()
    sar = torch.randn(1, 2, 16, 16)
    cloudy = torch.randn(1, 13, 16, 16)

    with torch.no_grad():
        _, sar_com, cld_com, sar_comp, cld_comp = model(sar, cloudy)

        sar_feat_l = model.sar_encoder(sar)
        cld_feat_l = model.cld_encoder(cloudy)
        sar_feat = model.up(sar_feat_l)
        cld_feat = model.up(cld_feat_l)

    assert torch.allclose(sar_comp, sar_feat - sar_com)
    assert torch.allclose(cld_comp, cld_feat - cld_com)


def test_resize_conv_up_uses_resize_and_64_channel_refine() -> None:
    up = ResizeConvUp(256).eval()
    feature = torch.randn(2, 256, 8, 8)

    with torch.no_grad():
        actual = up(feature)

    assert actual.shape == (2, 64, 16, 16)
    assert up.stem[0].in_channels == 256
    assert up.stem[0].out_channels == 64
    assert up.stem[0].kernel_size == (3, 3)
    assert len(up.refine) == 2
    assert not hasattr(up, "out_proj")
    assert not any(isinstance(module, nn.PixelShuffle) for module in up.modules())


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
