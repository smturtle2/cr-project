"""GPU 없이 더미 데이터로 모델/loss/optimizer가 정상 동작하는지 확인하는 테스트.

실제 데이터셋 다운로드나 cr-train 없이도 실행 가능하다.
Colab에 올리기 전에 코드 로직 자체를 먼저 검증하는 용도.

실행:
    python test_dummy.py
"""

from __future__ import annotations

import time

import torch

from modules.model import build_model
from modules.optimizer import build_optimizer
from modules.loss_fn import build_loss
from modules.metrics import build_metrics


def test_model_forward():
    """모델 forward pass 확인."""
    print("=" * 60)
    print("[1/5] 모델 forward pass 테스트")
    print("=" * 60)

    # SEN12MS-CR 입력 규격:
    #   SAR:    (B, 2, H, W)  - Sentinel-1 VV/VH
    #   Cloudy: (B, 13, H, W) - Sentinel-2 13 밴드
    #   Target: (B, 13, H, W) - 구름 없는 Sentinel-2

    batch_size = 2
    H, W = 64, 64  # CPU에서 빠른 테스트를 위해 작게

    # 작은 모델로 빠르게 테스트
    model = build_model(sar_channels=2, opt_channels=13, base_channels=16)
    model.eval()

    # 더미 입력 생성 (정규화된 값 범위: [0, 1])
    sar = torch.rand(batch_size, 2, H, W)
    cloudy = torch.rand(batch_size, 13, H, W)

    print(f"입력 shape:")
    print(f"  sar    : {tuple(sar.shape)}")
    print(f"  cloudy : {tuple(cloudy.shape)}")

    # Forward pass
    start = time.time()
    with torch.no_grad():
        prediction = model(sar, cloudy)
    elapsed = time.time() - start

    print(f"출력 shape: {tuple(prediction.shape)}")
    print(f"출력 범위: [{prediction.min():.4f}, {prediction.max():.4f}]")
    print(f"Forward 시간: {elapsed:.3f}s")

    # 출력 shape가 기대대로인지 확인
    expected_shape = (batch_size, 13, H, W)
    assert prediction.shape == expected_shape, \
        f"출력 shape가 다름: {prediction.shape} != {expected_shape}"

    # NaN/Inf 체크
    assert torch.isfinite(prediction).all(), "출력에 NaN/Inf가 있음"

    print("PASS: Forward 정상\n")
    return model, sar, cloudy


def test_loss_function(prediction_shape=(2, 13, 64, 64)):
    """Loss 함수 동작 확인."""
    print("=" * 60)
    print("[2/5] Loss 함수 테스트")
    print("=" * 60)

    loss_fn = build_loss(freq_weight=0.1, phase_weight=0.1)

    prediction = torch.rand(prediction_shape, requires_grad=True)
    target = torch.rand(prediction_shape)
    batch = {"target": target}

    loss = loss_fn(prediction, batch)
    print(f"Loss 값: {loss.item():.6f}")
    print(f"Loss shape: {loss.shape} (스칼라여야 함)")

    assert loss.dim() == 0, "loss는 스칼라여야 함"
    assert torch.isfinite(loss), "loss가 NaN/Inf"
    assert loss.item() > 0, "loss는 양수여야 함"

    # 역전파 가능한지 확인
    loss.backward()
    assert prediction.grad is not None, "gradient가 계산되지 않음"
    assert torch.isfinite(prediction.grad).all(), "gradient에 NaN/Inf"

    print(f"Gradient 범위: [{prediction.grad.min():.6f}, {prediction.grad.max():.6f}]")
    print("PASS: Loss + Backward 정상\n")


def test_metrics(prediction_shape=(2, 13, 64, 64)):
    """Metric 함수 동작 확인."""
    print("=" * 60)
    print("[3/5] Metrics 테스트")
    print("=" * 60)

    metrics = build_metrics()
    print(f"사용 metric: {list(metrics.keys())}")

    prediction = torch.rand(prediction_shape)
    target = torch.rand(prediction_shape)
    batch = {"target": target}

    for name, metric_fn in metrics.items():
        value = metric_fn(prediction, batch)
        assert value.dim() == 0, f"{name}은 스칼라여야 함"
        assert torch.isfinite(value), f"{name}이 NaN/Inf"
        print(f"  {name:6s}: {value.item():.6f}")

    # PSNR 상한값 체크 (동일한 텐서)
    psnr_fn = metrics["psnr"]
    identical_batch = {"target": prediction}
    psnr_val = psnr_fn(prediction, identical_batch)
    print(f"  동일 텐서 PSNR: {psnr_val.item():.2f} dB (매우 커야 함)")

    print("PASS: Metrics 정상\n")


def test_optimizer_step():
    """Optimizer로 1 step 학습이 되는지 확인."""
    print("=" * 60)
    print("[4/5] Optimizer 1-step 학습 테스트")
    print("=" * 60)

    model = build_model(sar_channels=2, opt_channels=13, base_channels=16)
    optimizer = build_optimizer(model, lr=1e-4, weight_decay=1e-4)
    loss_fn = build_loss(freq_weight=0.1)

    print(f"Optimizer: {type(optimizer).__name__}")
    print(f"LR: {optimizer.param_groups[0]['lr']}")
    print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    # 더미 배치
    sar = torch.rand(2, 2, 64, 64)
    cloudy = torch.rand(2, 13, 64, 64)
    target = torch.rand(2, 13, 64, 64)
    batch = {"target": target}

    # 학습 전 loss
    model.train()
    prediction = model(sar, cloudy)
    loss_before = loss_fn(prediction, batch).item()
    print(f"초기 loss: {loss_before:.6f}")

    # 5 step 학습
    for step in range(5):
        optimizer.zero_grad()
        prediction = model(sar, cloudy)
        loss = loss_fn(prediction, batch)
        loss.backward()
        optimizer.step()
        print(f"  step {step+1}: loss={loss.item():.6f}")

    loss_after = loss.item()
    print(f"감소량: {loss_before - loss_after:.6f}")

    assert loss_after < loss_before, "loss가 감소하지 않음 (학습 안 됨)"
    print("PASS: Optimizer 학습 정상\n")


def test_full_pipeline():
    """전체 파이프라인 mini-epoch 시뮬레이션."""
    print("=" * 60)
    print("[5/5] 전체 파이프라인 mini-epoch 시뮬레이션")
    print("=" * 60)

    model = build_model(sar_channels=2, opt_channels=13, base_channels=16)
    optimizer = build_optimizer(model)
    loss_fn = build_loss()
    metrics = build_metrics()

    # 더미 데이터셋: 10개 샘플, batch_size=2 -> 5 iteration
    num_samples = 10
    batch_size = 2
    H, W = 64, 64

    dummy_sars = torch.rand(num_samples, 2, H, W)
    dummy_cloudy = torch.rand(num_samples, 13, H, W)
    dummy_targets = torch.rand(num_samples, 13, H, W)

    print(f"더미 데이터셋: {num_samples}개 샘플, batch_size={batch_size}")

    model.train()
    start = time.time()
    total_loss = 0.0
    metric_sums = {name: 0.0 for name in metrics}

    for i in range(0, num_samples, batch_size):
        sar = dummy_sars[i:i+batch_size]
        cloudy = dummy_cloudy[i:i+batch_size]
        target = dummy_targets[i:i+batch_size]
        batch = {"target": target}

        optimizer.zero_grad()
        prediction = model(sar, cloudy)
        loss = loss_fn(prediction, batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            for name, fn in metrics.items():
                metric_sums[name] += fn(prediction, batch).item()

    num_batches = num_samples // batch_size
    elapsed = time.time() - start

    print(f"평균 loss: {total_loss/num_batches:.6f}")
    for name, total in metric_sums.items():
        print(f"평균 {name}: {total/num_batches:.6f}")
    print(f"총 소요 시간: {elapsed:.2f}s ({elapsed/num_batches:.2f}s/batch)")
    print("PASS: 전체 파이프라인 정상\n")


def main():
    print("\n" + "#" * 60)
    print("# cr-project 더미 테스트 (CPU only)")
    print("#" * 60 + "\n")

    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용: {torch.cuda.is_available()}")
    print(f"디바이스: CPU\n")

    try:
        model, sar, cloudy = test_model_forward()
        test_loss_function()
        test_metrics()
        test_optimizer_step()
        test_full_pipeline()

        print("=" * 60)
        print("모든 테스트 PASS - Colab에서 실행 준비 완료")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n[FAIL] 테스트 실패: {e}")
        raise
    except Exception as e:
        print(f"\n[ERROR] 예외 발생: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
