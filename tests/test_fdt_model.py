from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from modules.model.fdt import FDT, FDT_CRNet_Direct, FDT_CRNet_Side, ResizeConvUp
from modules.model.fdt_cca import (
    CCAMask,
    CCA_CRNet,
    Extractor,
    ExtractorLayer,
    FDT_CCA,
    FDT_CRNet_CCA,
    Stem,
)
from modules.model.module.attention import MultiHeadAttention


class RecordingBlock(nn.Module):
    call_shapes: list[tuple[int, tuple[int, int]]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.call_shapes.append((x.shape[1], tuple(x.shape[-2:])))
        return x

    @classmethod
    def reset(cls) -> None:
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


class FixedFeature(nn.Module):
    def __init__(self, feature: torch.Tensor):
        super().__init__()
        self.register_buffer("feature", feature)
        self.input: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.input = x.detach().clone()
        return self.feature.expand(x.shape[0], -1, x.shape[-2], x.shape[-1])


class InputCapture(nn.Module):
    def __init__(self):
        super().__init__()
        self.input: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.input = x.detach().clone()
        return x


class FixedMask(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, cld_cloud: torch.Tensor) -> torch.Tensor:
        return cld_cloud.new_full(
            (cld_cloud.shape[0], 1, cld_cloud.shape[-2], cld_cloud.shape[-1]),
            self.value,
        )


class ConstantImage(nn.Module):
    def __init__(self, out_channels: int, value: float):
        super().__init__()
        self.out_channels = out_channels
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_full(
            (x.shape[0], self.out_channels, x.shape[-2], x.shape[-1]),
            self.value,
        )


class AddFeatureDelta(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x.new_full(x.shape, self.value)


def test_fdt_imports_and_runs_forward() -> None:
    model = FDT(num_layers=1, num_heads=4).eval()
    assert isinstance(model.sar_encoder.proj[-1].proj[1], nn.PixelUnshuffle)
    assert model.sar_encoder.proj[-1].proj[0].out_channels == model.dim // 4
    assert isinstance(model.up.up.proj[1], nn.PixelShuffle)
    assert model.up.up.proj[0].out_channels == model.up_dim * 4
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
    assert isinstance(up.up.proj[1], nn.PixelShuffle)
    assert up.up.proj[0].out_channels == 256
    feature = torch.randn(2, 256, 8, 8)

    with torch.no_grad():
        actual = up(feature)

    assert actual.shape == (2, 64, 16, 16)
    assert actual.dtype == feature.dtype
    assert bool(torch.isfinite(actual).all().item())


def test_fdt_cca_returns_full_resolution_sar_clear_and_cloud_features() -> None:
    model = FDT_CCA(num_layers=1, num_heads=4).eval()
    sar = torch.randn(1, 2, 16, 16)
    cloudy = torch.randn(1, 13, 16, 16)

    with torch.no_grad():
        outputs = model(sar, cloudy)

    assert len(outputs) == 4
    assert outputs[0].shape == (1, 256, 16, 16)
    for output in outputs[1:]:
        assert output.shape == (1, 128, 16, 16)
        assert bool(torch.isfinite(output).all().item())


def test_stems_and_extractors_return_full_resolution_features() -> None:
    sar_stem = Stem(2, 128).eval()
    cld_stem = Stem(15, 128).eval()
    assert sar_stem.proj[0].kernel_size == (7, 7)
    assert sar_stem.proj[0].padding == (3, 3)
    assert sar_stem.proj[0].padding_mode == "reflect"
    assert cld_stem.proj[0].kernel_size == (7, 7)
    assert cld_stem.proj[0].padding == (3, 3)
    assert cld_stem.proj[0].padding_mode == "reflect"
    sar_extractor = Extractor(
        128,
        dims=(128, 256, 512),
        layer_count=2,
        num_layers=1,
        heads=4,
    ).eval()
    first_layer = sar_extractor.layers[0]
    assert isinstance(first_layer.down.downs[0].proj[1], nn.PixelUnshuffle)
    assert first_layer.down.downs[0].proj[0].out_channels == 64
    assert isinstance(first_layer.down.downs[1].proj[1], nn.PixelUnshuffle)
    assert first_layer.down.downs[1].proj[0].out_channels == 128
    assert isinstance(first_layer.up.ups[0].up.proj[1], nn.PixelShuffle)
    assert first_layer.up.ups[0].up.proj[0].out_channels == 1024
    assert isinstance(first_layer.up.ups[1].up.proj[1], nn.PixelShuffle)
    assert first_layer.up.ups[1].up.proj[0].out_channels == 512
    cld_extractor = Extractor(
        128,
        dims=(128, 256, 512),
        layer_count=2,
        num_layers=1,
        heads=4,
    ).eval()
    sar = torch.randn(1, 2, 16, 16)
    cloudy = torch.randn(1, 13, 16, 16)

    with torch.no_grad():
        sar_feat = sar_extractor(sar_stem(sar))
        cld_feat = cld_extractor(cld_stem(torch.cat((sar, cloudy), dim=1)))

    assert sar_feat.shape == (1, 128, 16, 16)
    assert cld_feat.shape == (1, 128, 16, 16)
    assert sar_feat.dtype == sar.dtype
    assert cld_feat.dtype == cloudy.dtype
    assert bool(torch.isfinite(sar_feat).all().item())
    assert bool(torch.isfinite(cld_feat).all().item())


def test_extractor_accepts_feature_inputs_for_clear_and_cloud_paths() -> None:
    extractor = Extractor(128, dims=(128, 256, 512), num_layers=1, heads=4).eval()
    sar_feat = torch.randn(1, 128, 16, 16)
    cld_feat = torch.randn(1, 128, 16, 16)

    with torch.no_grad():
        sar_feature = extractor(sar_feat)
        cld_clear = extractor(cld_feat)

    assert sar_feature.shape == sar_feat.shape
    assert cld_clear.shape == cld_feat.shape
    assert bool(torch.isfinite(sar_feature).all().item())
    assert bool(torch.isfinite(cld_clear).all().item())


def test_fdt_cca_uses_stem_features_for_extraction() -> None:
    model = FDT_CCA(
        dim=8,
        num_layers=1,
        num_heads=1,
        extractor_dims=(4, 8),
    ).eval()
    sar_stem_feat = torch.ones(1, 4, 1, 1)
    cld_stem_feat = torch.full((1, 4, 1, 1), 2.0)
    model.sar_stem = FixedFeature(sar_stem_feat)
    model.cld_stem = FixedFeature(cld_stem_feat)
    model.sar_extractor = InputCapture()
    model.cld_extractor = InputCapture()
    model.cld_clear_extractor = AddFeatureDelta(1.0)

    sar = torch.randn(1, 2, 1, 1)
    cloudy = torch.randn(1, 13, 1, 1)

    with torch.no_grad():
        outputs = model(sar, cloudy)

    assert torch.allclose(model.sar_stem.input, sar)
    assert torch.allclose(model.cld_stem.input, torch.cat((sar, cloudy), dim=1))
    assert torch.allclose(model.sar_extractor.input, sar_stem_feat)
    assert torch.allclose(model.cld_extractor.input, cld_stem_feat)
    assert torch.allclose(
        outputs[0],
        torch.cat((sar_stem_feat, cld_stem_feat + 1.0), dim=1),
    )
    assert torch.allclose(outputs[2], cld_stem_feat + 1.0)
    assert torch.allclose(outputs[3], cld_stem_feat - outputs[2])


def test_extractor_stacks_same_dim_layers() -> None:
    extractor = Extractor(
        4,
        dims=(4, 8),
        layer_count=2,
        num_layers=1,
        heads=1,
    ).eval()
    extractor.layers = nn.ModuleList([AddFeatureDelta(1.0), AddFeatureDelta(2.0)])
    feature = torch.zeros(1, 4, 2, 2)

    with torch.no_grad():
        actual = extractor(feature)

    assert torch.allclose(actual, torch.full_like(feature, 3.0))


def test_extractor_applies_blocks_only_on_top_down_levels() -> None:
    RecordingBlock.reset()
    extractor = ExtractorLayer(
        8,
        dims=(8, 16, 32),
        num_layers=1,
        heads=4,
    ).eval()
    extractor.up.blocks = nn.ModuleList([RecordingBlock(), RecordingBlock()])

    with torch.no_grad():
        extractor(torch.randn(1, 8, 16, 16))

    assert RecordingBlock.call_shapes == [(32, (4, 4)), (16, (8, 8))]


def test_extractor_uses_reconstruction_residual_skips() -> None:
    RecordingBlock.reset()
    ZeroRecon.reset()
    extractor = ExtractorLayer(
        8,
        dims=(8, 16, 32),
        num_layers=1,
        heads=4,
    ).eval()
    extractor.up.blocks = nn.ModuleList([RecordingBlock(), RecordingBlock()])
    extractor.up.recons = nn.ModuleList(
        [ZeroRecon(8, record=True), ZeroRecon(16, record=True)]
    )

    with torch.no_grad():
        extractor(torch.randn(1, 8, 16, 16))

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
    sar = torch.randn(1, 2, 32, 32)
    cloudy = torch.randn(1, 13, 32, 32)

    with torch.no_grad():
        outputs = model(sar, cloudy)

    assert outputs[0].shape == (1, 64, 32, 32)
    for output in outputs[1:]:
        assert output.shape == (1, 32, 32, 32)


def test_fdt_cca_rejects_invalid_extractor_config() -> None:
    with pytest.raises(ValueError, match="extractor_dims\\[0\\] \\* 2"):
        FDT_CCA(dim=256, extractor_dims=(64, 128))

    with pytest.raises(ValueError, match="at least two levels"):
        FDT_CCA(dim=256, extractor_dims=(128,))

    with pytest.raises(ValueError, match="divisible by heads"):
        FDT_CCA(dim=256, extractor_dims=(128, 258))

    with pytest.raises(ValueError, match="feature_extractor_layers"):
        FDT_CCA(feature_extractor_layers=0)


def test_cca_mask_returns_per_band_intervention_mask() -> None:
    cca = CCAMask(cloud_channels=128, mask_channels=13).eval()
    cld_cloud = torch.randn(2, 128, 16, 16)

    with torch.no_grad():
        mask = cca(cld_cloud)

    assert mask.shape == (2, 13, 16, 16)
    assert torch.all(mask >= 0.0)
    assert torch.all(mask <= 1.0)


def test_cca_mask_starts_with_small_uniform_mask() -> None:
    cca = CCAMask(cloud_channels=128, mask_channels=13).eval()
    cld_cloud = torch.randn(2, 128, 16, 16)
    expected = torch.full(
        (2, 13, 16, 16),
        torch.sigmoid(torch.tensor(-5.0)).item(),
    )

    with torch.no_grad():
        mask = cca(cld_cloud)

    assert torch.allclose(mask, expected)


def test_fdt_crnet_cca_keeps_small_mask_init_after_crnet_init() -> None:
    model = FDT_CRNet_CCA(fdt_layers=1, cr_layers=1).eval()
    cld_cloud = torch.randn(2, model.cloud_channels, 16, 16)
    expected = torch.full(
        (2, 13, 16, 16),
        torch.sigmoid(torch.tensor(-5.0)).item(),
    )

    with torch.no_grad():
        mask = model.crnet.cca(cld_cloud)

    assert torch.allclose(mask, expected)


def test_cca_crnet_runs_without_attention_layer() -> None:
    model = CCA_CRNet(out_channels=13, num_layers=0).eval()
    feature = torch.randn(1, 256, 16, 16)
    cld_cloud = torch.randn(1, 128, 16, 16)
    cloudy = torch.randn(1, 13, 16, 16)

    with torch.no_grad():
        prediction = model(feature, cld_cloud, cloudy)

    assert prediction.shape == (1, 13, 16, 16)
    assert prediction.dtype == feature.dtype


def test_cca_crnet_applies_masked_signed_delta_to_cloudy() -> None:
    model = CCA_CRNet(
        out_channels=1,
        num_layers=0,
        feature_sizes=2,
        cloud_channels=2,
        detail_blocks=1,
    ).eval()
    model.delta_head = ConstantImage(out_channels=1, value=-4.0)
    model.cca = FixedMask(value=0.25)
    feature = torch.zeros(1, 2, 4, 4)
    cld_cloud = torch.randn(1, 2, 4, 4)
    cloudy = torch.full((1, 1, 4, 4), 10.0)

    with torch.no_grad():
        prediction = model(feature, cld_cloud, cloudy)

    assert torch.allclose(prediction, torch.full_like(prediction, 9.0))


def test_cca_crnet_returns_candidate_and_mask_when_requested() -> None:
    model = CCA_CRNet(out_channels=13, num_layers=0).eval()
    feature = torch.randn(1, 256, 16, 16)
    cld_cloud = torch.randn(1, 128, 16, 16)
    cloudy = torch.randn(1, 13, 16, 16)

    with torch.no_grad():
        prediction, candidate, mask = model(
            feature,
            cld_cloud,
            cloudy,
            return_candidate=True,
            return_mask=True,
        )

    assert prediction.shape == cloudy.shape
    assert candidate.shape == cloudy.shape
    assert mask.shape == cloudy.shape
    assert torch.all(mask >= 0.0)
    assert torch.all(mask <= 1.0)


def test_cca_crnet_runs_with_delta_head() -> None:
    model = CCA_CRNet(out_channels=13, num_layers=4).eval()
    feature = torch.randn(1, 256, 16, 16)
    cld_cloud = torch.randn(1, 128, 16, 16)
    cloudy = torch.randn(1, 13, 16, 16)

    with torch.no_grad():
        prediction = model(feature, cld_cloud, cloudy)

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


def test_fdt_crnet_cca_returns_clear_cloud_contract() -> None:
    model = FDT_CRNet_CCA(fdt_layers=1, cr_layers=1, return_decomposition=True).eval()
    sar = torch.randn(1, 2, 16, 16)
    cloudy = torch.randn(1, 13, 16, 16)

    with torch.no_grad():
        outputs = model(sar, cloudy)

    assert len(outputs) == 6
    prediction, candidate, mask, sar_feat, cld_clear, cld_cloud = outputs
    assert prediction.shape == cloudy.shape
    assert candidate.shape == cloudy.shape
    assert mask.shape == cloudy.shape
    for feature in (sar_feat, cld_clear, cld_cloud):
        assert feature.shape == (1, 128, 16, 16)


def test_fdt_crnet_side_imports_and_runs_forward() -> None:
    model = FDT_CRNet_Side(fdt_layers=1, cr_layers=1).eval()
    sar = torch.randn(1, 2, 16, 16)
    cloudy = torch.randn(1, 13, 16, 16)

    with torch.no_grad():
        prediction = model(sar, cloudy)

    assert prediction.shape == cloudy.shape
    assert prediction.dtype == cloudy.dtype
