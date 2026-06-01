from __future__ import annotations

import ast
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from modules.loss_fn import CLEAR_NetLoss, make_clear_net_loss_fn
from modules.model.CLEAR_Net import (
    ACA_CRNet,
    CLEAR_Net,
    RefineHead,
    SampleDown,
    SampleUp,
    SpectralMaskRouter,
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


def test_sample_blocks_resize_and_project_without_attention_modules() -> None:
    down = SampleDown(4, 8).eval()
    up = SampleUp(8, 4).eval()
    x = torch.randn(1, 4, 8, 8)

    with torch.no_grad():
        low = down(x)
        high = up(low)

    assert low.shape == (1, 8, 4, 4)
    assert high.shape == x.shape
    module_names = {module.__class__.__name__ for module in [*down.modules(), *up.modules()]}
    assert "ChannelAttention" not in module_names
    assert "SpatialAttention" not in module_names


def test_spectral_mask_router_uses_opacity_route_and_channel_routes() -> None:
    router = SpectralMaskRouter(channels=4, out_channels=3).eval()
    x = torch.randn(2, 4, 5, 6)

    with torch.no_grad():
        mask = router(x)

    assert router.num_routes == 16
    assert not hasattr(router, "zero_route")
    assert isinstance(router.opacity_head, RefineHead)
    assert isinstance(router.route_head, RefineHead)
    assert router.channel_routes.weight.shape == (16, 3)
    assert mask.shape == (2, 3, 5, 6)
    assert torch.all(mask >= 0.0)
    assert torch.all(mask <= 1.0)


def test_clear_net_owns_feature_paths_directly() -> None:
    model = CLEAR_Net(
        dim=8,
        feature_layers=1,
        extractor_layers=1,
        cr_layers=0,
        num_heads=1,
        extractor_dims=(4, 8),
        return_decomposition=True,
    ).eval()
    sar = torch.randn(1, 2, 8, 8)
    cloudy = torch.randn(1, 13, 8, 8)

    with torch.no_grad():
        outputs = model(sar, cloudy)

    assert hasattr(model, "sar_stem")
    assert hasattr(model, "cloudy_stem")
    assert hasattr(model, "clear_extractor")
    assert hasattr(model, "aux_head")
    assert isinstance(model.aux_head, RefineHead)
    assert isinstance(model.aca_crnet.candidate_head, RefineHead)
    assert isinstance(model.aca_crnet.mask_router, SpectralMaskRouter)
    assert not any(isinstance(module, nn.Sigmoid) for module in model.aca_crnet.modules())
    sar_feat = outputs["sar_feat"]
    clear_feat = outputs["clear_feat"]
    cloud_feat = outputs["cloud_feat"]
    aux_clear = outputs["aux_clear"]
    assert sar_feat.shape == (1, 4, 8, 8)
    assert clear_feat.shape == (1, 4, 8, 8)
    assert cloud_feat.shape == (1, 4, 8, 8)
    assert aux_clear.shape == cloudy.shape
    assert bool(torch.isfinite(clear_feat).all().item())


def test_clear_net_uses_contiguous_conv_inputs() -> None:
    model = CLEAR_Net(
        dim=8,
        feature_layers=1,
        extractor_layers=1,
        cr_layers=2,
        num_heads=1,
        extractor_dims=(4, 8),
        return_decomposition=True,
    ).eval()
    hits = []

    def record_non_contiguous_input(name: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            x = inputs[0]
            if x.dim() == 4 and not x.is_contiguous():
                hits.append(name)

        return hook

    handles = [
        module.register_forward_pre_hook(record_non_contiguous_input(name))
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]
    sar = torch.randn(1, 2, 8, 8)
    cloudy = torch.randn(1, 13, 8, 8)

    try:
        with torch.no_grad():
            model(sar, cloudy)
    finally:
        for handle in handles:
            handle.remove()

    assert hits == []


def test_clear_net_defaults_use_half_width() -> None:
    model = CLEAR_Net(
        feature_layers=1,
        extractor_layers=1,
        cr_layers=0,
        return_decomposition=True,
    ).eval()
    sar = torch.randn(1, 2, 8, 8)
    cloudy = torch.randn(1, 13, 8, 8)

    with torch.no_grad():
        outputs = model(sar, cloudy)

    assert model.dim == 128
    assert model.extractor_dims == (64, 128, 256)
    assert model.fused_extractor_dims == (128, 256, 512)
    assert model.feature_channels == 64
    assert outputs["prediction"].shape == cloudy.shape
    assert outputs["sar_feat"].shape == (1, 64, 8, 8)
    assert outputs["clear_feat"].shape == (1, 64, 8, 8)


def test_aca_crnet_blends_cloudy_with_raw_candidate() -> None:
    model = ACA_CRNet(out_channels=1, num_layers=0, feature_sizes=2, cloud_channels=2).eval()
    model.candidate_head = ConstantImage(out_channels=1, value=8.0)
    mask_value = 0.25
    model.mask_router = ConstantImage(out_channels=1, value=mask_value)
    fused = torch.zeros(1, 2, 4, 4)
    cloud_feat = torch.randn(1, 2, 4, 4)
    cloudy = torch.full((1, 1, 4, 4), 1.0)

    with torch.no_grad():
        output = model(fused, cloud_feat, cloudy)

    prediction = output["prediction"]
    candidate = output["candidate"]
    mask = output["mask"]
    expected_candidate = torch.full_like(candidate, 8.0)
    expected_mask = torch.full_like(mask, mask_value)
    expected_prediction = cloudy * (1.0 - mask) + expected_candidate * mask
    assert torch.allclose(candidate, expected_candidate)
    assert torch.allclose(mask, expected_mask)
    assert torch.allclose(prediction, expected_prediction)


def test_clear_net_forward_and_decomposition_contract() -> None:
    model = CLEAR_Net(
        dim=8,
        feature_layers=1,
        extractor_layers=1,
        cr_layers=0,
        num_heads=1,
        extractor_dims=(4, 8),
        return_decomposition=True,
    ).eval()
    sar = torch.randn(1, 2, 8, 8)
    cloudy = torch.randn(1, 13, 8, 8)

    with torch.no_grad():
        outputs = model(sar, cloudy)

    assert set(outputs) == {
        "prediction",
        "candidate",
        "mask",
        "sar_feat",
        "clear_feat",
        "cloud_feat",
        "aux_clear",
    }
    prediction = outputs["prediction"]
    candidate = outputs["candidate"]
    mask = outputs["mask"]
    sar_feat = outputs["sar_feat"]
    clear_feat = outputs["clear_feat"]
    cloud_feat = outputs["cloud_feat"]
    aux_clear = outputs["aux_clear"]
    assert prediction.shape == cloudy.shape
    assert candidate.shape == cloudy.shape
    assert mask.shape == cloudy.shape
    assert aux_clear.shape == cloudy.shape
    for feature in (sar_feat, clear_feat, cloud_feat):
        assert feature.shape == (1, 4, 8, 8)
    assert torch.all(mask >= 0.0)
    assert torch.all(mask <= 1.0)
    assert bool(torch.isfinite(candidate).all().item())
    assert bool(torch.isfinite(aux_clear).all().item())


def test_clear_net_loss_combines_l1_and_ssim() -> None:
    loss_fn = CLEAR_NetLoss()
    prediction = torch.rand(2, 13, 16, 16) * 5.0
    target = torch.rand(2, 13, 16, 16) * 5.0
    model_output = {"prediction": prediction}

    loss = loss_fn(model_output, target)
    expected = F.l1_loss(prediction, target) + 0.1 * (
        1.0 - loss_fn.ssim(prediction, target)
    )

    assert torch.allclose(loss, expected)


def test_clear_net_loss_adds_candidate_and_aux_reconstruction_losses() -> None:
    loss_fn = CLEAR_NetLoss()
    prediction = torch.rand(2, 13, 16, 16) * 5.0
    candidate = torch.rand(2, 13, 16, 16) * 5.0
    aux_prediction = torch.rand(2, 13, 16, 16) * 5.0
    target = torch.rand(2, 13, 16, 16) * 5.0
    model_output = {
        "prediction": prediction,
        "candidate": candidate,
        "aux_clear": aux_prediction,
    }

    loss = loss_fn(model_output, target)
    expected = (
        F.l1_loss(prediction, target)
        + 0.1 * loss_fn.ssim_loss(prediction, target)
        + 0.5
        * (
            F.l1_loss(candidate, target)
            + 0.1 * loss_fn.ssim_loss(candidate, target)
        )
        + 0.2
        * (
            F.l1_loss(aux_prediction, target)
            + 0.1 * loss_fn.ssim_loss(aux_prediction, target)
        )
    )

    assert torch.allclose(loss, expected)


def test_clear_net_loss_candidate_weight_can_be_disabled() -> None:
    loss_fn = CLEAR_NetLoss(candidate_weight=0.0, aux_weight=0.0)
    prediction = torch.zeros(1, 1, 4, 4)
    target = torch.zeros_like(prediction)
    candidate = torch.ones_like(prediction)
    model_output = {
        "prediction": prediction,
        "candidate": candidate,
    }

    loss = loss_fn(model_output, target)
    expected = loss_fn.reconstruction_loss(prediction, target)

    assert torch.allclose(loss, expected)


def test_clear_net_loss_factory_accepts_training_batch_contract() -> None:
    loss_fn = make_clear_net_loss_fn()
    criterion = CLEAR_NetLoss()
    prediction = torch.rand(2, 13, 16, 16) * 5.0
    target = torch.rand(2, 13, 16, 16) * 5.0

    loss = loss_fn({"prediction": prediction}, {"target": target})

    assert torch.allclose(loss, criterion(prediction, target))


def test_clear_net_ssim_data_range_matches_five_unit_inputs() -> None:
    loss_fn = CLEAR_NetLoss()
    unit_range_loss_fn = CLEAR_NetLoss(data_range=1.0)
    prediction = torch.rand(2, 13, 16, 16) * 5.0
    target = torch.rand(2, 13, 16, 16) * 5.0

    ssim = loss_fn.ssim(prediction, target)
    expected = unit_range_loss_fn.ssim(prediction / 5.0, target / 5.0)

    assert torch.allclose(ssim, expected, atol=1e-6)


def test_clear_net_package_has_no_legacy_module_imports() -> None:
    root = Path(__file__).resolve().parents[1] / "modules" / "model" / "CLEAR_Net"
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [] if node.module is None else [node.module]
            else:
                continue
            assert not any(name == "modules" or name.startswith("modules.") for name in imported)
