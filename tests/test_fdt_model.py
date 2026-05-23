from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from modules.model.fdt import FDT, FDT_CRNet_Direct, FDT_CRNet_Side, ResizeConvUp
from modules.model.fdt_cca import (
    CCA_AttnAdapter,
    CCA_CRNet,
    Extractor,
    FDT_CCA,
    FDT_CRNet_CCA,
    JointBlock,
    ParallelBlock,
)
from modules.model.module.attention import MultiHeadAttention


class RecordingBlock(nn.Module):
    init_dims: list[int] = []
    call_shapes: list[tuple[int, tuple[int, int]]] = []

    def __init__(self, dim: int, num_layers: int, heads: int):
        super().__init__()
        self.init_dims.append(dim)

    def forward(
        self,
        sar: torch.Tensor,
        cld: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.call_shapes.append((sar.shape[1], tuple(sar.shape[-2:])))
        return sar, cld

    @classmethod
    def reset(cls) -> None:
        cls.init_dims = []
        cls.call_shapes = []


class ZeroRecon(nn.Module):
    recon_shapes: list[tuple[int, tuple[int, int], tuple[int, int]]] = []

    def __init__(
        self,
        out_channels: int,
        scale_factor: int = 2,
        record: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.record = record

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height = x.shape[-2] * self.scale_factor
        width = x.shape[-1] * self.scale_factor
        if self.record:
            self.recon_shapes.append(
                (x.shape[1], tuple(x.shape[-2:]), (height, width))
            )
        return x.new_zeros((x.shape[0], self.out_channels, height, width))

    @classmethod
    def reset(cls) -> None:
        cls.recon_shapes = []


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


def test_extractor_returns_full_resolution_features() -> None:
    extractor = Extractor(
        2,
        13,
        dims=(128, 256, 512),
        num_layers=1,
        heads=4,
        block_cls=ParallelBlock,
    ).eval()
    sar = torch.randn(1, 2, 16, 16)
    cloudy = torch.randn(1, 13, 16, 16)

    with torch.no_grad():
        sar_feat, cld_feat = extractor(sar, cloudy)

    assert sar_feat.shape == (1, 128, 16, 16)
    assert cld_feat.shape == (1, 128, 16, 16)
    assert sar_feat.dtype == sar.dtype
    assert cld_feat.dtype == cloudy.dtype
    assert bool(torch.isfinite(sar_feat).all().item())
    assert bool(torch.isfinite(cld_feat).all().item())


def test_extractor_accepts_feature_inputs_for_common() -> None:
    extractor = Extractor(
        128,
        128,
        dims=(128, 256, 512),
        num_layers=1,
        heads=4,
        block_cls=JointBlock,
    ).eval()
    sar_feat = torch.randn(1, 128, 16, 16)
    cld_feat = torch.randn(1, 128, 16, 16)

    with torch.no_grad():
        sar_com, cld_com = extractor(sar_feat, cld_feat)

    assert sar_com.shape == sar_feat.shape
    assert cld_com.shape == cld_feat.shape
    assert bool(torch.isfinite(sar_com).all().item())
    assert bool(torch.isfinite(cld_com).all().item())


def test_extractor_applies_blocks_only_on_top_down_levels() -> None:
    RecordingBlock.reset()
    extractor = Extractor(
        2,
        13,
        dims=(8, 16, 32),
        num_layers=1,
        heads=4,
        block_cls=RecordingBlock,
    ).eval()

    with torch.no_grad():
        extractor(torch.randn(1, 2, 16, 16), torch.randn(1, 13, 16, 16))

    assert RecordingBlock.init_dims == [32, 16]
    assert RecordingBlock.call_shapes == [(32, (4, 4)), (16, (8, 8))]


def test_extractor_uses_reconstruction_residual_skips() -> None:
    RecordingBlock.reset()
    ZeroRecon.reset()
    extractor = Extractor(
        2,
        13,
        dims=(8, 16, 32),
        num_layers=1,
        heads=4,
        block_cls=RecordingBlock,
    ).eval()
    extractor.up.sar_recons = nn.ModuleList(
        [ZeroRecon(8, record=True), ZeroRecon(16, record=True)]
    )
    extractor.up.cld_recons = nn.ModuleList([ZeroRecon(8), ZeroRecon(16)])

    with torch.no_grad():
        extractor(torch.randn(1, 2, 16, 16), torch.randn(1, 13, 16, 16))

    assert ZeroRecon.recon_shapes == [
        (16, (8, 8), (16, 16)),
        (32, (4, 4), (8, 8)),
    ]


def test_fdt_cca_accepts_two_level_extractor() -> None:
    model = FDT_CCA(
        dim=256,
        num_layers=1,
        num_heads=4,
        extractor_dims=(128, 256),
    ).eval()
    sar = torch.randn(1, 2, 16, 16)
    cloudy = torch.randn(1, 13, 16, 16)

    with torch.no_grad():
        outputs = model(sar, cloudy)

    assert outputs[0].shape == (1, 256, 16, 16)
    for output in outputs[1:]:
        assert output.shape == (1, 128, 16, 16)


def test_fdt_cca_accepts_four_level_extractor() -> None:
    model = FDT_CCA(
        dim=64,
        num_layers=1,
        num_heads=4,
        extractor_dims=(32, 64, 128, 256),
    ).eval()
    sar = torch.randn(1, 2, 16, 16)
    cloudy = torch.randn(1, 13, 16, 16)

    with torch.no_grad():
        outputs = model(sar, cloudy)

    assert outputs[0].shape == (1, 64, 16, 16)
    for output in outputs[1:]:
        assert output.shape == (1, 32, 16, 16)


def test_fdt_cca_rejects_invalid_extractor_config() -> None:
    with pytest.raises(ValueError, match="extractor_dims\\[0\\] \\* 2"):
        FDT_CCA(dim=256, extractor_dims=(64, 128))

    with pytest.raises(ValueError, match="at least two levels"):
        FDT_CCA(dim=256, extractor_dims=(128,))

    with pytest.raises(ValueError, match="divisible by heads"):
        FDT_CCA(dim=256, extractor_dims=(128, 258))


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
