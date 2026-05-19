from __future__ import annotations

import torch
import torch.nn as nn

from modules.model.fdt import FDT, FDT_CRNet_Direct, FDT_CRNet_Side, ResizeConvUp
from modules.model.fdt_mask import FDTMask, FDT_CRNet_Mask, MaskEncoder
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


def test_fdt_decomposes_sar_and_cloudy_features() -> None:
    model = FDT(num_layers=1, num_heads=4).eval()
    assert hasattr(model, "sar_common_encoder")
    assert hasattr(model, "cld_common_encoder")
    assert not hasattr(model, "joint_encoder")
    assert not hasattr(model, "feat1_encoder")
    assert not hasattr(model, "feat2_encoder")

    sar = torch.randn(1, 2, 16, 16)
    cloudy = torch.randn(1, 13, 16, 16)

    with torch.no_grad():
        fdt_feature, sar_com, cld_com, sar_comp, cld_comp = model(sar, cloudy)
        sar_feat = model.up(model.sar_encoder(sar))
        cld_feat = model.up(model.cld_encoder(cloudy))
        com_fused = model.com_fuse(torch.cat((sar_com, cld_com), dim=1))

    assert torch.allclose(sar_comp, sar_feat - sar_com)
    assert torch.allclose(cld_comp, cld_feat - cld_com)
    assert torch.allclose(fdt_feature[:, :128], com_fused)
    assert torch.allclose(fdt_feature[:, 128:192], sar_comp)
    assert torch.allclose(fdt_feature[:, 192:], cld_comp)


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


def test_fdt_mask_returns_full_resolution_cloudy_component_only() -> None:
    model = FDTMask(num_layers=1, num_heads=4).eval()
    sar = torch.randn(1, 2, 16, 16)
    cloudy = torch.randn(1, 13, 16, 16)

    with torch.no_grad():
        outputs = model(sar, cloudy)

    assert len(outputs) == 5
    assert outputs[0].shape == (1, 256, 16, 16)
    for output in outputs[1:]:
        assert output.shape == (1, 128, 16, 16)
        assert bool(torch.isfinite(output).all().item())


def test_mask_encoder_downsamples_full_resolution_cloudy_component() -> None:
    encoder = MaskEncoder(dim=256, out_channels=13, num_layers=1, heads=4).eval()
    cld_comp = torch.randn(2, 128, 16, 16)

    with torch.no_grad():
        mask = encoder(cld_comp)

    assert mask.shape == (2, 13, 16, 16)
    assert mask.dtype == cld_comp.dtype
    assert bool(torch.isfinite(mask).all().item())
    assert float(mask.min()) >= 0.0
    assert float(mask.max()) <= 1.0


def test_fdt_crnet_direct_imports_and_runs_forward() -> None:
    model = FDT_CRNet_Direct(fdt_layers=1, cr_layers=1).eval()
    sar = torch.randn(1, 2, 16, 16)
    cloudy = torch.randn(1, 13, 16, 16)

    with torch.no_grad():
        prediction = model(sar, cloudy)

    assert prediction.shape == cloudy.shape
    assert prediction.dtype == cloudy.dtype


def test_fdt_crnet_mask_imports_and_runs_forward() -> None:
    model = FDT_CRNet_Mask(fdt_layers=1, cr_layers=1).eval()
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
