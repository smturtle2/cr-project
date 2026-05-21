from __future__ import annotations

import torch
import torch.nn as nn

from modules.model.fdt import FDT, FDT_CRNet_Direct, FDT_CRNet_Side, ResizeConvUp
from modules.model.fdt_cca import (
    CCA_AttnAdapter,
    CCA_CRNet,
    FDT_CCA,
    FDT_CRNet_CCA,
)
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


def test_multi_head_attention_accepts_distinct_value_source() -> None:
    attn = MultiHeadAttention(dim=4, num_heads=1).eval()
    for layer in (attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj):
        nn.init.eye_(layer.weight)
        nn.init.zeros_(layer.bias)

    query = torch.zeros(1, 2, 4)
    key_source = torch.randn(1, 2, 4)
    value_source = torch.tensor(
        [[[1.0, 3.0, 5.0, 7.0], [9.0, 11.0, 13.0, 15.0]]]
    )

    with torch.no_grad():
        actual = attn(query, key_source, value_source)

    expected = value_source.mean(dim=1, keepdim=True).expand_as(actual)
    assert torch.allclose(actual, expected)


def test_resize_conv_up_returns_expected_feature_shape() -> None:
    up = ResizeConvUp(256).eval()
    feature = torch.randn(2, 256, 8, 8)

    with torch.no_grad():
        actual = up(feature)

    assert actual.shape == (2, 64, 16, 16)
    assert actual.dtype == feature.dtype
    assert bool(torch.isfinite(actual).all().item())


def test_fdt_cca_returns_full_resolution_cloudy_component_only() -> None:
    model = FDT_CCA(num_layers=1, num_heads=4).eval()
    sar = torch.randn(1, 2, 16, 16)
    cloudy = torch.randn(1, 13, 16, 16)

    with torch.no_grad():
        outputs = model(sar, cloudy)

    assert len(outputs) == 5
    assert outputs[0].shape == (1, 256, 16, 16)
    for output in outputs[1:]:
        assert output.shape == (1, 128, 16, 16)
        assert bool(torch.isfinite(output).all().item())


def test_cca_attn_adapter_starts_as_identity() -> None:
    adapter = CCA_AttnAdapter(
        comp_channels=128,
        feature_channels=256,
        num_layers=1,
        heads=4,
    ).eval()
    feature = torch.randn(2, 256, 16, 16)
    cld_comp = torch.randn(2, 128, 16, 16)

    with torch.no_grad():
        actual = adapter(feature, cld_comp)

    assert torch.allclose(actual, feature)


def test_cca_crnet_runs_without_attention_layer() -> None:
    model = CCA_CRNet(out_channels=13, num_layers=0).eval()
    feature = torch.randn(1, 256, 16, 16)
    cld_comp = torch.randn(1, 128, 16, 16)

    with torch.no_grad():
        prediction = model(feature, cld_comp)

    assert prediction.shape == (1, 13, 16, 16)
    assert prediction.dtype == feature.dtype


def test_cca_crnet_runs_with_adapter_injection() -> None:
    model = CCA_CRNet(out_channels=13, num_layers=4).eval()
    feature = torch.randn(1, 256, 16, 16)
    cld_comp = torch.randn(1, 128, 16, 16)

    with torch.no_grad():
        prediction = model(feature, cld_comp)

    assert prediction.shape == (1, 13, 16, 16)
    assert prediction.dtype == feature.dtype


def test_fdt_crnet_direct_imports_and_runs_forward() -> None:
    model = FDT_CRNet_Direct(fdt_layers=1, cr_layers=1).eval()
    sar = torch.randn(1, 2, 16, 16)
    cloudy = torch.randn(1, 13, 16, 16)

    with torch.no_grad():
        prediction = model(sar, cloudy)

    assert prediction.shape == cloudy.shape
    assert prediction.dtype == cloudy.dtype


def test_fdt_crnet_cca_imports_and_runs_forward() -> None:
    model = FDT_CRNet_CCA(fdt_layers=1, cr_layers=1).eval()
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
