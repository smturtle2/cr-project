from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from modules.loss_fn import CLEAR_NetLoss, make_clear_net_loss_fn
from modules.model.CLEAR_Net import (
    ACA_CRNet,
    CLEAR_Net,
    CLEAR_Net_New,
    ConAttn,
    Extractor,
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


class ConstantMaskRouter(nn.Module):
    def __init__(self, out_channels: int, value: float):
        super().__init__()
        self.out_channels = out_channels
        self.value = value

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mask = x.new_full(
            (x.shape[0], self.out_channels, x.shape[-2], x.shape[-1]),
            self.value,
        )
        route_weights = x.new_full((x.shape[0], 1, x.shape[-2], x.shape[-1]), 1.0)
        return {
            "mask": mask,
            "route_weights": route_weights,
        }


class InputCapture(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.input = x
        return x


class AddFeatureDelta(nn.Module):
    def __init__(self, delta: float):
        super().__init__()
        self.delta = delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.delta


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


def test_spectral_mask_router_uses_routed_channel_logits() -> None:
    router = SpectralMaskRouter(channels=4, out_channels=3).eval()
    x = torch.randn(2, 4, 5, 6)

    with torch.no_grad():
        output = router(x)

    parameters = dict(router.named_parameters())
    mask = output["mask"]
    route_weights = output["route_weights"]
    assert router.num_routes == 64
    assert not hasattr(router, "zero_route")
    assert not hasattr(router, "opacity_head")
    assert isinstance(router.route_head, RefineHead)
    assert router.channel_routes.shape == (64, 3)
    assert parameters["channel_routes"] is router.channel_routes
    assert mask.shape == (2, 3, 5, 6)
    assert route_weights.shape == (2, 64, 5, 6)
    assert torch.allclose(route_weights.sum(dim=1), torch.ones_like(route_weights[:, 0]))
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
    assert hasattr(model, "cld_sar_stem")
    assert hasattr(model, "cld_hsi_stem")
    assert hasattr(model, "cld_extractor")
    assert hasattr(model, "cld_clean_extractor")
    assert hasattr(model, "cld_cloudy_extractor")
    assert hasattr(model, "fused_extractor")
    assert hasattr(model, "fused_refiner")
    assert hasattr(model, "aux_head")
    assert model.cld_stem_channels == 2
    assert model.cld_sar_stem.proj[0].out_channels == 2
    assert model.cld_hsi_stem.proj[0].out_channels == 2
    assert model.fused_extractor_dims == (8, 16)
    assert isinstance(model.fused_extractor, Extractor)
    assert isinstance(model.fused_refiner, RefineHead)
    assert isinstance(model.aux_head, RefineHead)
    assert isinstance(model.aca_crnet.candidate_head, RefineHead)
    assert isinstance(model.mask_router, SpectralMaskRouter)
    assert model.mask_router.num_routes == 64
    assert not hasattr(model.aca_crnet, "mask_router")
    assert not any(isinstance(module, nn.Sigmoid) for module in model.aca_crnet.modules())
    sar_feat = outputs["sar_feat"]
    cld_feat = outputs["cld_feat"]
    cld_clean = outputs["cld_clean"]
    cld_cloudy = outputs["cld_cloudy"]
    aux_clear = outputs["aux_clear"]
    assert sar_feat.shape == (1, 4, 8, 8)
    assert cld_feat.shape == (1, 4, 8, 8)
    assert cld_clean.shape == (1, 4, 8, 8)
    assert cld_cloudy.shape == (1, 4, 8, 8)
    assert aux_clear.shape == cloudy.shape
    assert bool(torch.isfinite(cld_clean).all().item())


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
    assert model.extractor_dims == (64, 128, 160)
    assert model.fused_extractor_dims == (128, 256, 320)
    assert isinstance(model.fused_extractor, Extractor)
    assert isinstance(model.fused_refiner, RefineHead)
    assert model.feature_channels == 64
    assert model.cld_stem_channels == 32
    assert outputs["prediction"].shape == cloudy.shape
    assert outputs["sar_feat"].shape == (1, 64, 8, 8)
    assert outputs["cld_clean"].shape == (1, 64, 8, 8)


def test_clear_net_new_defaults_use_extractor_only_width() -> None:
    model = CLEAR_Net_New(
        feature_layers=1,
        extractor_layers=0,
        candidate_extractor_passes=0,
        return_decomposition=True,
    ).eval()
    sar = torch.randn(1, 2, 8, 8)
    cloudy = torch.randn(1, 13, 8, 8)

    with torch.no_grad():
        outputs = model(sar, cloudy)

    assert model.dim == 32
    assert model.extractor_dims == (32, 128, 160)
    assert model.feature_channels == 32
    assert model.fused_channels == 64
    assert model.cld_stem_channels == 16
    assert model.candidate_extractor_passes == 0
    assert not hasattr(model, "aca_crnet")
    assert not hasattr(model, "aux_head")
    assert isinstance(model.fused_refiner, RefineHead)
    assert isinstance(model.candidate_extractor, Extractor)
    assert isinstance(model.candidate_head, RefineHead)
    assert outputs["prediction"].shape == cloudy.shape
    assert outputs["sar_feat"].shape == (1, 32, 8, 8)
    assert outputs["cld_clean"].shape == (1, 32, 8, 8)


def test_clear_net_new_forward_and_decomposition_contract() -> None:
    model = CLEAR_Net_New(
        feature_layers=1,
        extractor_layers=1,
        candidate_extractor_passes=2,
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
        "route_weights",
        "sar_feat",
        "cld_feat",
        "cld_clean",
        "cld_cloudy",
    }
    assert not any(isinstance(module, ACA_CRNet) for module in model.modules())
    assert model.fused_refiner.proj.in_channels == 8
    assert model.fused_refiner.proj.out_channels == 4
    assert len(model.candidate_extractor.layers) == 2
    assert outputs["prediction"].shape == cloudy.shape
    assert outputs["candidate"].shape == cloudy.shape
    assert outputs["mask"].shape == cloudy.shape
    assert outputs["route_weights"].shape == (1, 64, 8, 8)
    for feature in (
        outputs["sar_feat"],
        outputs["cld_feat"],
        outputs["cld_clean"],
        outputs["cld_cloudy"],
    ):
        assert feature.shape == (1, 4, 8, 8)
    assert torch.all(outputs["mask"] >= 0.0)
    assert torch.all(outputs["mask"] <= 1.0)
    assert torch.allclose(
        outputs["route_weights"].sum(dim=1),
        torch.ones_like(outputs["route_weights"][:, 0]),
    )
    assert bool(torch.isfinite(outputs["candidate"]).all().item())


def test_clear_net_extracts_fused_features_before_refine() -> None:
    model = CLEAR_Net(
        dim=8,
        feature_layers=1,
        extractor_layers=1,
        cr_layers=0,
        num_heads=1,
        extractor_dims=(4, 8),
        return_decomposition=True,
    ).eval()
    model.fused_extractor = AddFeatureDelta(1.0)
    model.fused_refiner = InputCapture()
    sar = torch.randn(1, 2, 8, 8)
    cloudy = torch.randn(1, 13, 8, 8)

    with torch.no_grad():
        outputs = model(sar, cloudy)

    expected_refiner_input = torch.cat(
        (outputs["sar_feat"], outputs["cld_clean"]),
        dim=1,
    ) + 1.0
    assert torch.allclose(model.fused_refiner.input, expected_refiner_input)


def test_aca_crnet_returns_candidate_only() -> None:
    model = ACA_CRNet(out_channels=1, num_layers=0, feature_sizes=2).eval()
    model.candidate_head = ConstantImage(out_channels=1, value=8.0)
    fused = torch.zeros(1, 2, 4, 4)

    with torch.no_grad():
        candidate = model(fused)

    expected_candidate = torch.full_like(candidate, 8.0)
    assert torch.allclose(candidate, expected_candidate)
    assert not isinstance(candidate, dict)


def test_clear_net_owns_mask_routing_and_final_blending() -> None:
    model = CLEAR_Net(
        dim=4,
        feature_layers=1,
        extractor_layers=1,
        cr_layers=0,
        num_heads=1,
        extractor_dims=(2, 4),
        return_decomposition=True,
    ).eval()
    model.aca_crnet = ConstantImage(out_channels=13, value=8.0)
    mask_value = 0.25
    model.mask_router = ConstantMaskRouter(out_channels=13, value=mask_value)
    sar = torch.zeros(1, 2, 8, 8)
    cloudy = torch.full((1, 13, 8, 8), 1.0)

    with torch.no_grad():
        output = model(sar, cloudy)

    expected_candidate = torch.full_like(output["candidate"], 8.0)
    expected_mask = torch.full_like(output["mask"], mask_value)
    expected_prediction = cloudy * (1.0 - expected_mask) + expected_candidate * expected_mask
    assert torch.allclose(output["candidate"], expected_candidate)
    assert torch.allclose(output["mask"], expected_mask)
    assert output["route_weights"].shape == (1, 1, 8, 8)
    assert torch.allclose(output["prediction"], expected_prediction)


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
        "route_weights",
        "sar_feat",
        "cld_feat",
        "cld_clean",
        "cld_cloudy",
        "aux_clear",
    }
    prediction = outputs["prediction"]
    candidate = outputs["candidate"]
    mask = outputs["mask"]
    route_weights = outputs["route_weights"]
    sar_feat = outputs["sar_feat"]
    cld_feat = outputs["cld_feat"]
    cld_clean = outputs["cld_clean"]
    cld_cloudy = outputs["cld_cloudy"]
    aux_clear = outputs["aux_clear"]
    assert prediction.shape == cloudy.shape
    assert candidate.shape == cloudy.shape
    assert mask.shape == cloudy.shape
    assert route_weights.shape == (1, 64, 8, 8)
    assert aux_clear.shape == cloudy.shape
    for feature in (sar_feat, cld_feat, cld_clean, cld_cloudy):
        assert feature.shape == (1, 4, 8, 8)
    assert torch.all(mask >= 0.0)
    assert torch.all(mask <= 1.0)
    assert torch.allclose(route_weights.sum(dim=1), torch.ones_like(route_weights[:, 0]))
    assert bool(torch.isfinite(candidate).all().item())
    assert bool(torch.isfinite(aux_clear).all().item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FlashAttention")
def test_clear_net_flash_conattn_cuda_forward_backward() -> None:
    model = ConAttn(
        input_channels=128,
        output_channels=128,
        num_heads=4,
        flash_dtype=torch.float16,
    ).cuda().half()
    x = torch.randn(1, 128, 16, 16, device="cuda", dtype=torch.float16, requires_grad=True)

    y = model(x)
    loss = y.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()

    assert y.shape == x.shape
    assert bool(torch.isfinite(y).all().item())
    assert x.grad is not None
    assert bool(torch.isfinite(x.grad).all().item())


def test_clear_net_flash_conattn_gate_scale_is_capped_and_suppressible() -> None:
    model = ConAttn(
        input_channels=8,
        output_channels=8,
        num_heads=1,
        lambda_init=1e-3,
        lambda_max=0.1,
    )

    assert torch.allclose(model._gate_scale(), torch.tensor(1e-3), atol=1e-7)

    with torch.no_grad():
        model.lambda_logit.fill_(-80.0)
    assert model._gate_scale().item() < 1e-6

    with torch.no_grad():
        model.lambda_logit.fill_(80.0)
    assert model._gate_scale().item() <= model.lambda_max + 1e-6
    assert model._gate_scale().item() > 0.099


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
        + F.l1_loss(candidate, target)
        + 0.1 * loss_fn.ssim_loss(candidate, target)
        + 0.1
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


def test_clear_net_loss_ignores_model_mask_for_reconstruction_loss() -> None:
    loss_fn = CLEAR_NetLoss(ssim_weight=0.0)
    prediction = torch.tensor([[[[1.0, 2.0]]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    model_mask = torch.tensor([[[[1.0, 0.0]]]], requires_grad=True)
    model_output = {
        "prediction": prediction,
        "mask": model_mask,
    }

    loss = loss_fn(model_output, target)

    expected = F.l1_loss(prediction, target)
    assert torch.allclose(loss, expected)
    loss.backward()
    assert prediction.grad is not None
    assert model_mask.grad is None


def test_clear_net_loss_adds_route_balance_loss() -> None:
    loss_fn = CLEAR_NetLoss(route_balance_weight=0.05)
    prediction = torch.zeros(2, 1, 4, 4)
    target = torch.zeros_like(prediction)
    route_weights = torch.zeros(2, 4, 4, 4)
    route_weights[:, 0] = 1.0
    model_output = {
        "prediction": prediction,
        "route_weights": route_weights,
    }

    loss = loss_fn(model_output, target)
    route_count = route_weights.size(1)
    router_prob_per_route = route_weights.mean(dim=(0, 2, 3))
    selected_routes = route_weights.argmax(dim=1)
    route_usage = F.one_hot(selected_routes, num_classes=route_count).float()
    route_usage = route_usage.mean(dim=(0, 1, 2)).detach()
    expected = loss_fn.reconstruction_loss(prediction, target)
    expected = expected + 0.05 * route_count * (route_usage * router_prob_per_route).sum()

    assert torch.allclose(loss, expected)


def test_clear_net_loss_default_route_balance_weight_is_two_thousandths() -> None:
    loss_fn = CLEAR_NetLoss()

    assert loss_fn.route_balance_weight == 0.002


def test_clear_net_loss_factory_accepts_training_batch_contract() -> None:
    loss_fn = make_clear_net_loss_fn()
    criterion = CLEAR_NetLoss()
    prediction = torch.rand(2, 13, 16, 16) * 5.0
    target = torch.rand(2, 13, 16, 16) * 5.0

    loss = loss_fn({"prediction": prediction}, {"target": target})

    assert torch.allclose(loss, criterion(prediction, target))


def test_clear_net_loss_factory_ignores_extra_batch_inputs() -> None:
    loss_fn = make_clear_net_loss_fn(
        ssim_weight=0.0,
    )
    criterion = CLEAR_NetLoss(
        ssim_weight=0.0,
    )
    prediction = torch.tensor([[[[1.0, 2.0]]]])
    target = torch.zeros_like(prediction)
    model_output = {
        "prediction": prediction,
    }

    loss = loss_fn(model_output, {"target": target, "cloudy": torch.ones_like(target)})

    expected = criterion(model_output, target)
    assert torch.allclose(loss, expected)


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
