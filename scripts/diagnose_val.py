"""val_loss 변동성 진단 스크립트.

체크:
  1. train/val 해상도 차이로 인한 metric 차이
  2. augmentation 전역 상태 누수
  3. 같은 val 이미지 2회 forward 시 재현성 (determinism)
  4. val set에 강한 outlier 있는지
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from modules.model.cafm.ACA_CRNet import ACA_CRNet
import tmp_main as tm


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # 같은 구성으로 모델 생성
    model = ACA_CRNet(
        sar_channels=2, opt_channels=13,
        num_layers=16, feature_sizes=256,
        use_cafm=True, cafm_feat_dim=32, use_sdi=True,
    ).to(device).eval()

    # 더미 입력 (256×256 val size) — batch=1 for OOM 회피
    sar_full = torch.rand(1, 2, 256, 256, device=device)
    cloudy_full = torch.rand(1, 13, 256, 256, device=device)
    target_full = torch.rand(1, 13, 256, 256, device=device)

    # 더미 입력 (128×128 train size)
    sar_crop = sar_full[:, :, :128, :128]
    cloudy_crop = cloudy_full[:, :, :128, :128]
    target_crop = target_full[:, :, :128, :128]

    print("=" * 60)
    print("[1] 256×256 (val style) vs 128×128 (train style) 출력 비교")
    print("=" * 60)
    with torch.no_grad():
        out_full = model(sar_full, cloudy_full)
        out_crop = model(sar_crop, cloudy_crop)
    mse_full = ((out_full - target_full) ** 2).mean().item()
    mse_crop = ((out_crop - target_crop) ** 2).mean().item()
    print(f"  256×256: MSE={mse_full:.6f}, out.shape={tuple(out_full.shape)}")
    print(f"  128×128: MSE={mse_crop:.6f}, out.shape={tuple(out_crop.shape)}")

    print("\n[2] 같은 입력 2회 forward — 결정성 체크")
    with torch.no_grad():
        o1 = model(sar_full, cloudy_full)
        o2 = model(sar_full, cloudy_full)
    diff = (o1 - o2).abs().max().item()
    print(f"  max |o1 - o2| = {diff:.2e} {'✅ 결정적' if diff < 1e-6 else '⚠️ 비결정적'}")

    print("\n[3] _aug_params 전역 상태 추적")
    import random as _r
    _r.seed(123)
    # train 모드에서 1회 forward (aug 적용)
    model.train()
    from tmp_main import enable_gradient_checkpointing
    # 이미 패치됐을 수 있으므로 새 인스턴스
    model2 = ACA_CRNet(
        sar_channels=2, opt_channels=13,
        num_layers=16, feature_sizes=256,
        use_cafm=True, cafm_feat_dim=32, use_sdi=True,
    ).to(device)
    enable_gradient_checkpointing(model2)
    print(f"  before train forward: _aug_params = {tm._aug_params}")

    model2.train()
    _ = model2(sar_full, cloudy_full)
    print(f"  after train forward:  _aug_params = {tm._aug_params}")

    model2.eval()
    _ = model2(sar_full, cloudy_full)
    print(f"  after eval forward:   _aug_params = {tm._aug_params}")

    print("\n[4] density estimator의 256 vs 128 민감도")
    with torch.no_grad():
        d_full = model.density_estimator(sar_full, cloudy_full)
        d_crop = model.density_estimator(sar_crop, cloudy_crop)
    print(f"  256×256 density: mean={d_full.mean().item():.4f} std={d_full.std().item():.4f}")
    print(f"  128×128 density: mean={d_crop.mean().item():.4f} std={d_crop.std().item():.4f}")

    print("\n[5] CAFM 변조 차이 (같은 feature, 같은 density mean)")
    feat_test = torch.randn(2, 256, 32, 32, device=device)
    density_test_low = torch.full((2, 1, 32, 32), 0.3, device=device)
    density_test_high = torch.full((2, 1, 32, 32), 0.7, device=device)
    with torch.no_grad():
        out_low = model.cafm1(feat_test, density_test_low)
        out_high = model.cafm1(feat_test, density_test_high)
    delta = (out_high - out_low).abs().mean().item()
    print(f"  CAFM(density=0.3) vs CAFM(density=0.7) 평균 차이: {delta:.6f}")
    if delta < 1e-4:
        print("  ⚠️ CAFM이 density에 거의 반응 안 함 (zero-init 덜 풀림)")


if __name__ == "__main__":
    main()
