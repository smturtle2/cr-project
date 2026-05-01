"""SDI 구현 검증 스크립트 (학습 전 sanity checks)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from modules.model.cafm.ACA_CRNet import ACA_CRNet


def make_dummy(B=2, H=64, W=64, device="cpu"):
    torch.manual_seed(42)
    sar = torch.rand(B, 2, H, W, device=device)
    cloudy = torch.rand(B, 13, H, W, device=device)
    return sar, cloudy


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def build(use_cafm, use_sdi):
    torch.manual_seed(0)
    return ACA_CRNet(
        sar_channels=2, opt_channels=13,
        num_layers=16, feature_sizes=256,
        use_cafm=use_cafm, use_sdi=use_sdi,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}")
    sar, cloudy = make_dummy(device=device)
    results = {}

    # ─────────────────────────────────────────────────
    # 1. Import & instantiation
    # ─────────────────────────────────────────────────
    print("\n[1] Import & instantiation")
    m_base = build(use_cafm=False, use_sdi=False).to(device).eval()
    m_cafm = build(use_cafm=True,  use_sdi=False).to(device).eval()
    m_sdi  = build(use_cafm=False, use_sdi=True ).to(device).eval()
    m_both = build(use_cafm=True,  use_sdi=True ).to(device).eval()
    assert hasattr(m_sdi, "sar_translator")
    assert hasattr(m_sdi, "sdi_injector")
    assert not hasattr(m_base, "sar_translator")
    print("  OK — attrs correct")

    # ─────────────────────────────────────────────────
    # 2. Shape test
    # ─────────────────────────────────────────────────
    print("\n[2] Shape test")
    with torch.no_grad():
        for name, m in [("base", m_base), ("cafm", m_cafm),
                        ("sdi", m_sdi), ("both", m_both)]:
            out = m(sar, cloudy)
            assert out.shape == cloudy.shape, f"{name}: {out.shape}"
            print(f"  {name}: out.shape={tuple(out.shape)} OK")

    # ─────────────────────────────────────────────────
    # 3. Parameter count
    # ─────────────────────────────────────────────────
    print("\n[3] Parameter counts")
    n_base = count_params(m_base)
    n_cafm = count_params(m_cafm)
    n_sdi  = count_params(m_sdi)
    n_both = count_params(m_both)
    d_sdi = n_sdi - n_base
    print(f"  base:       {n_base:>12,}")
    print(f"  +CAFM:      {n_cafm:>12,}  (+{n_cafm-n_base:,})")
    print(f"  +SDI:       {n_sdi:>12,}  (+{d_sdi:,})")
    print(f"  +CAFM+SDI:  {n_both:>12,}  (+{n_both-n_base:,})")
    # SDI 예상: Conv(15,32)≈4.3K + 4×Resblock(~18.5K) + Conv(32,13)≈3.7K + λ(1) ≈ 82K
    assert 60_000 < d_sdi < 120_000, f"SDI param delta {d_sdi} out of range"
    print("  OK — SDI delta in [60K, 120K]")

    # ─────────────────────────────────────────────────
    # 4. Identity test — λ=0 이면 use_sdi ON/OFF 동일 출력
    # ─────────────────────────────────────────────────
    print("\n[4] Identity test (λ=0)")
    # 같은 init seed 필요 — 공통 backbone만 복사
    m_base_ref = build(use_cafm=False, use_sdi=False).to(device).eval()
    m_sdi_test = build(use_cafm=False, use_sdi=True ).to(device).eval()
    # backbone 가중치를 m_base_ref에서 m_sdi_test로 복사
    base_sd = m_base_ref.state_dict()
    sdi_sd = m_sdi_test.state_dict()
    for k, v in base_sd.items():
        if k in sdi_sd:
            sdi_sd[k].copy_(v)
    m_sdi_test.load_state_dict(sdi_sd, strict=False)

    with torch.no_grad():
        out_base = m_base_ref(sar, cloudy)
        out_sdi  = m_sdi_test(sar, cloudy)
    max_diff = (out_base - out_sdi).abs().max().item()
    print(f"  max |out_base - out_sdi(λ=0)| = {max_diff:.2e}")
    assert max_diff < 1e-5, f"IDENTITY VIOLATED: {max_diff}"
    print("  OK — SDI is identity at λ=0")

    # CAFM + SDI 조합도 동일 체크
    m_cafm_ref = build(use_cafm=True,  use_sdi=False).to(device).eval()
    m_both_test = build(use_cafm=True,  use_sdi=True ).to(device).eval()
    cafm_sd = m_cafm_ref.state_dict()
    both_sd = m_both_test.state_dict()
    for k, v in cafm_sd.items():
        if k in both_sd:
            both_sd[k].copy_(v)
    m_both_test.load_state_dict(both_sd, strict=False)
    with torch.no_grad():
        out_cafm = m_cafm_ref(sar, cloudy)
        out_both = m_both_test(sar, cloudy)
    max_diff2 = (out_cafm - out_both).abs().max().item()
    print(f"  max |out_cafm - out_both(λ=0)| = {max_diff2:.2e}")
    assert max_diff2 < 1e-5, f"IDENTITY (CAFM+SDI) VIOLATED: {max_diff2}"
    print("  OK — CAFM+SDI is identity at λ=0")

    # ─────────────────────────────────────────────────
    # 5. Density gating branch
    # ─────────────────────────────────────────────────
    print("\n[5] Density gating branch")
    # λ를 강제로 0.5로 만들어 효과 확인
    with torch.no_grad():
        m_sdi.sdi_injector.lam.fill_(0.5)
        m_both.sdi_injector.lam.fill_(0.5)
        out_sdi_inj  = m_sdi(sar, cloudy)           # density=None
        out_both_inj = m_both(sar, cloudy)          # density is used
    assert (out_sdi_inj - m_base(sar, cloudy)).abs().max() > 1e-4, \
        "no injection effect in +SDI path"
    assert (out_both_inj - m_cafm(sar, cloudy)).abs().max() > 1e-4, \
        "no injection effect in +CAFM+SDI path"
    # 리셋
    with torch.no_grad():
        m_sdi.sdi_injector.lam.zero_()
        m_both.sdi_injector.lam.zero_()
    print("  OK — both paths inject when λ≠0")

    # ─────────────────────────────────────────────────
    # 6. Gradient flow
    # ─────────────────────────────────────────────────
    print("\n[6] Gradient flow")
    m_both.train()
    m_both.sdi_injector.lam.data.fill_(0.1)  # 0이면 lam.grad가 product rule로 non-trivial하지만 translator쪽은 0
    out = m_both(sar, cloudy)
    loss = out.mean()
    loss.backward()
    # translator gradient 확인
    none_grads = []
    nan_grads = []
    for name, p in m_both.named_parameters():
        if p.grad is None:
            none_grads.append(name)
        elif torch.isnan(p.grad).any():
            nan_grads.append(name)
    assert not none_grads, f"None grads: {none_grads[:3]}"
    assert not nan_grads, f"NaN grads: {nan_grads[:3]}"
    assert m_both.sdi_injector.lam.grad is not None
    assert m_both.sar_translator.conv1.weight.grad is not None
    print(f"  all {len(list(m_both.named_parameters()))} params have grads, no NaN")
    print(f"  lam.grad = {m_both.sdi_injector.lam.grad.item():.4e}")
    print(f"  translator.conv1 grad norm = "
          f"{m_both.sar_translator.conv1.weight.grad.norm().item():.4e}")

    # ─────────────────────────────────────────────────
    # 7. λ learning check (3 optimizer steps)
    # ─────────────────────────────────────────────────
    print("\n[7] λ moves under optimizer")
    m_step = build(use_cafm=True, use_sdi=True).to(device).train()
    opt = torch.optim.AdamW(m_step.parameters(), lr=1e-3)
    target = torch.rand_like(cloudy)
    lam_hist = [m_step.sdi_injector.lam.item()]
    for _ in range(3):
        opt.zero_grad()
        out = m_step(sar, cloudy)
        ((out - target) ** 2).mean().backward()
        opt.step()
        lam_hist.append(m_step.sdi_injector.lam.item())
    print(f"  λ trajectory: {[f'{v:.4e}' for v in lam_hist]}")
    assert abs(lam_hist[-1]) > 0, "λ did not move"
    print("  OK — λ updates")

    # ─────────────────────────────────────────────────
    # 8. last_pseudo_opt stored
    # ─────────────────────────────────────────────────
    print("\n[8] last_pseudo_opt attribute")
    with torch.no_grad():
        m_sdi.sdi_injector.lam.fill_(0.1)
        _ = m_sdi(sar, cloudy)
    assert m_sdi.last_pseudo_opt is not None
    assert m_sdi.last_pseudo_opt.shape == cloudy.shape
    print(f"  last_pseudo_opt.shape = {tuple(m_sdi.last_pseudo_opt.shape)} OK")

    # ─────────────────────────────────────────────────
    # 9. checkpointed_forward (tmp_main.py) 경로 검증
    # ─────────────────────────────────────────────────
    print("\n[9] checkpointed_forward path (tmp_main.py)")
    from torch.utils.checkpoint import checkpoint as ckpt

    def checkpointed_forward(model, sar, cloudy):
        # tmp_main.py의 enable_gradient_checkpointing 복제
        model.last_cloudy = cloudy
        if model.use_cafm:
            model.last_density = model.density_estimator(sar, cloudy)
        else:
            model.last_density = None
        x = torch.cat([cloudy, sar], dim=1)
        feat = model.head(x)
        feat = ckpt(model.body1, feat, use_reentrant=False)
        if model.use_cafm:
            feat = model.cafm1(feat, model.last_density)
        feat = ckpt(model.body2, feat, use_reentrant=False)
        if model.use_cafm:
            feat = model.cafm2(feat, model.last_density)
        feat = model.body3(feat)
        out = model.tail(feat)
        base_pred = cloudy + out
        if model.use_sdi:
            pseudo_opt = model.sar_translator(cloudy, sar)
            model.last_pseudo_opt = pseudo_opt
            pred = model.sdi_injector(
                base_pred, pseudo_opt, density=model.last_density,
            )
        else:
            model.last_pseudo_opt = None
            pred = base_pred
        return pred

    m_ckpt = build(use_cafm=True, use_sdi=True).to(device).train()
    out_ckpt = checkpointed_forward(m_ckpt, sar, cloudy)
    assert out_ckpt.shape == cloudy.shape
    loss = out_ckpt.mean()
    loss.backward()
    assert m_ckpt.sdi_injector.lam.grad is not None
    assert m_ckpt.sar_translator.conv1.weight.grad is not None
    print(f"  checkpointed forward OK, shape={tuple(out_ckpt.shape)}")
    print(f"  lam.grad = {m_ckpt.sdi_injector.lam.grad.item():.4e}")

    # λ=0에서 checkpointed forward도 base와 동일한지 (CAFM+SDI 경로)
    m_ckpt_base = build(use_cafm=True, use_sdi=False).to(device).eval()
    m_ckpt_sdi = build(use_cafm=True, use_sdi=True).to(device).eval()
    # weight 복사
    base_sd2 = m_ckpt_base.state_dict()
    sdi_sd2 = m_ckpt_sdi.state_dict()
    for k, v in base_sd2.items():
        if k in sdi_sd2:
            sdi_sd2[k].copy_(v)
    m_ckpt_sdi.load_state_dict(sdi_sd2, strict=False)
    with torch.no_grad():
        out_c_base = checkpointed_forward(m_ckpt_base, sar, cloudy)
        out_c_sdi = checkpointed_forward(m_ckpt_sdi, sar, cloudy)
    diff_c = (out_c_base - out_c_sdi).abs().max().item()
    print(f"  max |ckpt_base - ckpt_sdi(λ=0)| = {diff_c:.2e}")
    assert diff_c < 1e-5, f"IDENTITY (checkpointed) VIOLATED: {diff_c}"
    print("  OK — checkpointed forward is identity at λ=0")

    # ─────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("ALL CHECKS PASSED ✅")
    print("=" * 50)


if __name__ == "__main__":
    main()
