from __future__ import annotations

import torch
import torch.nn as nn

from modules.model.fdt import FDT, FDT_CRNet_Direct, FDT_CRNet_Side, ResizeConvUp


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


def test_fdt_uses_paired_common_transformer() -> None:
    model = FDT(num_layers=1, num_heads=4).eval()

    assert not hasattr(model, "sar_common_gate")
    assert not hasattr(model, "cld_common_gate")
    assert model.common_extractor.num_layers == model.num_layers
    assert len(model.common_extractor.blocks) == model.num_layers

    block = model.common_extractor.blocks[0]
    assert block.heads == model.num_heads
    assert block.sar_self_attn is not block.cld_self_attn
    assert block.sar_cross_attn is not block.cld_cross_attn
    assert isinstance(block.sar_mlp, nn.Sequential)
    assert isinstance(block.cld_mlp, nn.Sequential)
    assert not hasattr(block, "num_experts")
    assert not hasattr(block, "experts")
    assert not hasattr(block, "router")


def test_common_extractor_returns_modality_specific_features() -> None:
    torch.manual_seed(0)
    model = FDT(num_layers=1, num_heads=4).eval()
    sar_feat = torch.randn(2, model.dim, 8, 8)
    cld_feat = torch.randn_like(sar_feat)

    with torch.no_grad():
        sar_com, cld_com = model.common_extractor(sar_feat, cld_feat)

    assert sar_com.shape == sar_feat.shape
    assert cld_com.shape == cld_feat.shape
    assert sar_com.dtype == sar_feat.dtype
    assert cld_com.dtype == cld_feat.dtype
    assert bool(torch.isfinite(sar_com).all().item())
    assert bool(torch.isfinite(cld_com).all().item())
    assert not torch.allclose(sar_com, cld_com)


def test_common_block_cross_reads_self_snapshots() -> None:
    class SelfMarker(nn.Module):
        def __init__(self, value: float):
            super().__init__()
            self.value = value

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.full_like(x, self.value)

    class CrossRecorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = None
            self.source = None

        def forward(
            self,
            query: torch.Tensor,
            tgt: torch.Tensor | None = None,
        ) -> torch.Tensor:
            self.query = query.detach().clone()
            self.source = tgt.detach().clone()
            return torch.zeros_like(query)

    class ZeroMLP(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

    model = FDT(dim=8, num_layers=1, num_heads=4).eval()
    block = model.common_extractor.blocks[0]
    block.sar_self_norm = nn.Identity()
    block.cld_self_norm = nn.Identity()
    block.sar_cross_query_norm = nn.Identity()
    block.sar_cross_source_norm = nn.Identity()
    block.cld_cross_query_norm = nn.Identity()
    block.cld_cross_source_norm = nn.Identity()
    block.sar_self_attn = SelfMarker(3.0)
    block.cld_self_attn = SelfMarker(5.0)
    block.sar_cross_attn = CrossRecorder()
    block.cld_cross_attn = CrossRecorder()
    block.sar_mlp_norm = nn.Identity()
    block.cld_mlp_norm = nn.Identity()
    block.sar_mlp = ZeroMLP()
    block.cld_mlp = ZeroMLP()

    sar_tokens = torch.zeros(1, 2, 8)
    cld_tokens = torch.zeros_like(sar_tokens)
    block(sar_tokens, cld_tokens)

    assert torch.allclose(block.sar_cross_attn.query, torch.full_like(sar_tokens, 3.0))
    assert torch.allclose(block.sar_cross_attn.source, torch.full_like(cld_tokens, 5.0))
    assert torch.allclose(block.cld_cross_attn.query, torch.full_like(cld_tokens, 5.0))
    assert torch.allclose(block.cld_cross_attn.source, torch.full_like(sar_tokens, 3.0))


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
