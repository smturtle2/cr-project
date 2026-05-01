"""CPU 진단 — ConAttn 및 aug 상태만 체크 (메모리 안전)."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import tmp_main as tm
from tmp_main import enable_gradient_checkpointing
from modules.model.cafm.ca import ConAttn


def main():
    torch.manual_seed(0)
    print("=" * 60)
    print("[A] ConAttn 128 vs 256 output statistics")
    print("=" * 60)
    ca = ConAttn(input_channels=32, output_channels=32, stride=4).eval()
    x128 = torch.randn(1, 32, 32, 32)
    x256 = torch.randn(1, 32, 64, 64)  # 2x spatial

    with torch.no_grad():
        y128 = ca(x128)
        y256 = ca(x256)
    print(f"  input  64×64:  out mean={y128.mean().item():.4f} std={y128.std().item():.4f}")
    print(f"  input 128×128: out mean={y256.mean().item():.4f} std={y256.std().item():.4f}")
    print(f"  ratio: std_256/std_128 = {y256.std().item()/y128.std().item():.3f}")
    print()
    print("  → attention output distribution이 해상도별로 크게 다르면")
    print("    ConAttn이 train(128) / val(256) 에서 다른 regime으로 동작")

    print("\n" + "=" * 60)
    print("[B] _aug_params 상태 전이 추적")
    print("=" * 60)
    from modules.model.cafm.ACA_CRNet import ACA_CRNet
    model = ACA_CRNet(
        sar_channels=2, opt_channels=13,
        num_layers=4, feature_sizes=32,  # 작게 만들어서 CPU로도 돌리기
        use_cafm=True, cafm_feat_dim=16, use_sdi=True,
    )
    enable_gradient_checkpointing(model)

    sar = torch.rand(1, 2, 256, 256)
    cloudy = torch.rand(1, 13, 256, 256)

    print(f"  초기:                      _aug_params = {tm._aug_params}")
    model.train()
    _ = model(sar, cloudy)
    print(f"  train() forward 직후:      _aug_params = {tm._aug_params[:4] if tm._aug_params else None}...")
    model.eval()
    _ = model(sar, cloudy)
    print(f"  eval() forward 직후:       _aug_params = {tm._aug_params}")

    # 만약 train → eval 전환 시 leak 있는지 (중간에 직접 안 초기화)
    model.train()
    _ = model(sar, cloudy)
    print(f"  다시 train() forward 직후: _aug_params = {tm._aug_params[:4] if tm._aug_params else None}...")
    # 이 상태에서 바로 loss를 호출하면 (시뮬레이션)
    target = torch.rand(1, 13, 256, 256)
    transformed = tm.transform_target(target)
    print(f"  transform_target 결과 shape: {transformed.shape} (예상: (1,13,128,128))")
    model.eval()
    _ = model(sar, cloudy)
    transformed2 = tm.transform_target(target)
    print(f"  eval forward 후 transform_target: {transformed2.shape} (예상: (1,13,256,256))")

    print("\n" + "=" * 60)
    print("[C] 같은 입력 2회 forward (eval) 결정성")
    print("=" * 60)
    model.eval()
    with torch.no_grad():
        o1 = model(sar, cloudy)
        o2 = model(sar, cloudy)
    diff = (o1 - o2).abs().max().item()
    status = "✅ 결정적" if diff < 1e-6 else "⚠️ 비결정적"
    print(f"  max |o1 - o2| = {diff:.2e}  {status}")


if __name__ == "__main__":
    main()
