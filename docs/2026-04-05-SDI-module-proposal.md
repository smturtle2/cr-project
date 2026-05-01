# SDI (SAR-Guided Detail Injection) 모듈 구현 보고서

**작성일**: 2026-04-05
**대상 베이스라인**: `ACA_CRNet + CAFM` (`modules/model/cafm/`)
**근거 논문**: Vo & Lee, *"Unrolled Low-Rank Tensor Completion for SAR-Guided Cloud Removal in Hyperspectral Images"*, IEEE TGRS 2025
**참조 코드**: `docs/SAR-ULRTC-CR/` (저자 공식 구현)

---

## 1. 배경 및 동기

### 베이스라인의 한계
현재 `ACA_CRNet`은 SAR(2ch)과 Optical(13ch)을 단순히 concat한 후 같은 `Conv3x3`로 처리 (`ACA_CRNet.py:218-219`):
```python
x = torch.cat([cloudy, sar], dim=1)  # (B, 15, H, W)
feat = self.head(x)                   # 공유 Conv
```
→ SAR 정보가 **15채널 중 2채널 지분**으로 섞여 과소 활용됨. 두꺼운 구름 영역에서 네트워크가 참고할 단서가 실질적으로 부족.

### SAR-Guided Detail Injection의 개념
SAR은 마이크로파로 구름을 투과하여 지표 구조(건물, 도로, 지형)를 관측 가능. 이를 **먼저 optical 도메인으로 번역**하여 pseudo-optical 이미지 `X̂`를 만들고, 이를 네트워크 출력의 **explicit detail prior**로 주입.

### 논문 내 기여도
- **Table IV (CHVo 2025)**: DI 제거 시 PSNR **33.81 → 32.11 (−1.7 dB)**, SSIM 0.954 → 0.948
- 논문의 4대 ablation 요소(LR / R / S / DI) 중 **두 번째로 큰 단일 기여**

---

## 2. 논문 원 구현 분석 (`docs/SAR-ULRTC-CR/main_net.py:11037-11091`)

놀랍게도 **F_Trans 자체는 모델 밖에 있습니다**:

```python
class RPCA_Net_4_stages_SAR_Trans_RGB_8(nn.Module):
    def __init__(self, N_iter):
        ...
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)  # lifting only

    def forward(self, RSI, SAR, trans):       # trans: 오프라인 pseudo-RGB
        cs_comp = self.dup_conv(trans)        # 3ch RGB → 13ch HSI
        ...
        for i in range(self.N_iter):
            ... = self.network[i](..., cs_comp)  # 각 stage에 주입
```

**요약**:
- SAR → RGB(3ch) 번역은 **SEN1-2 pretrained CycleGAN**으로 오프라인 처리 (`dataLoader.py:2118`, `train_translated/*.png`로 저장)
- 모델 내부의 translator 파라미터는 **1×1 Conv 1개 (39 params)** 뿐
- detail injection의 실체는 **"프리컴퓨트된 pseudo-optical을 매 stage에 더하는 것"**

---

## 3. 본 프로젝트 구현 전략

### 옵션 비교

| 옵션 | 설명 | 장점 | 단점 |
|---|---|---|---|
| **A. Paper-faithful (offline)** | 외부 CycleGAN 가중치 다운로드 → 전체 데이터셋 SAR 번역 → PNG 저장 → DataLoader에 필드 추가 | 논문 그대로 재현 | 전처리 pipeline 추가, 외부 가중치 필요, `cr-train` DataLoader 수정 필요 |
| **B. End-to-end (권장)** | 경량 SAR→13ch U-Net을 `ACA_CRNet` 내부에 포함, end-to-end 공동 학습 | 기존 pipeline 무변경, 이식 단순, 단일 프로세스 | translator가 명시적 supervision 없이 학습됨 |
| **C. Hybrid (다음 단계)** | B + translator 출력에 **보조 손실** (cloud-free 영역 target L1) | 가장 강한 gain 기대 | 구현 복잡도 ↑, loss 튜닝 필요 |

**권장**: **B를 먼저 구현 → 효과 확인 시 C로 확장**.
2048 샘플 regime에서는 translator에 보조 supervision을 주는 C가 결국 유리할 가능성이 크나, B로 빠르게 검증 후 진행.

---

## 4. 제안 아키텍처 (Option B)

### 데이터 흐름

```
sar (B,2,H,W)              cloudy (B,13,H,W)
    │                             │
    ▼                             │
┌────────────────┐                │
│  SARTranslator │  ★ 신규         │
│  (경량 U-Net)  │                │
└────────┬───────┘                │
         ▼                         │
   X̂ (B,13,H,W)                   │
   pseudo-optical                  │
         │                         │
         ├──────┐         ┌────────┘
         │      │         │
         │      │         ▼
         │      │  ┌─────────────────┐
         │      │  │ concat(cloudy,  │  (기존)
         │      │  │   sar) → head   │
         │      │  │   → body1       │
         │      │  │   → CAFM1       │
         │      │  │   → body2       │
         │      │  │   → CAFM2       │
         │      │  │   → body3       │
         │      │  │   → tail        │
         │      │  └────────┬────────┘
         │      │           │
         │      │           ▼ out (B,13,H,W)
         │      │           │
         │  ┌───┴───┐       │
         │  │proj2  │       │
         │  │1×1    │       │
         │  │13→13  │       │
         │  └───┬───┘       │
         │      │           │
         │      └─ λ_mid ──►│  (선택: body2 후 feature에도 주입)
         │                  │
         │                  ▼
         │             ┌─────────┐
         │             │   add   │
         │             └────┬────┘
         │                  │
         │        λ_out     │
         └─────────────────►│
                            ▼
                      ┌──────────┐
                      │  + cloudy│  (기존 residual)
                      └─────┬────┘
                            ▼
                          pred
```

### 수식
```
X̂       = SARTranslator(sar)                      # pseudo-optical (B,13,H,W)
feat     = body3(...(body1(head([cloudy, sar]))))  # 기존 경로
out      = tail(feat)
pred     = cloudy + out + λ_out · X̂                # SDI residual
```

### Mask 재활용 (CAFM과의 시너지)
CAFM이 이미 estimate한 `last_density`(픽셀별 구름 밀도 ∈[0,1])를 **SDI 주입 영역 게이팅**에 재활용:

```python
# 구름 영역에서만 X̂ 주입 (clean 영역은 건드리지 않음)
pred = cloudy + out + λ_out * last_density * X̂
```

이렇게 하면 **SDI × CAFM의 진정한 시너지**: CAFM은 feature 변조, SDI는 output-space residual, 그리고 둘 다 같은 density map을 공유.

---

## 5. 모듈 파일 구성

### 신규 파일: `modules/model/cafm/sdi.py`

```python
"""SDI — SAR-Guided Detail Injection module.

논문: Vo & Lee, TGRS 2025, "Unrolled LRTC for SAR-Guided Cloud Removal in HSIs"
     Sec III-B eq.(8), Table IV (DI ablation: +1.7 dB PSNR).

구조: 경량 U-Net으로 SAR(2ch) → pseudo-optical(13ch) 변환 후,
     ACA_CRNet의 output residual에 (선택적 density gating으로) 주입.
"""
from __future__ import annotations
import torch
from torch import nn


def _conv_bn_relu(in_c: int, out_c: int, k: int = 3, s: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, stride=s, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class SARTranslator(nn.Module):
    """SAR(2ch) → pseudo-optical(13ch).

    경량 U-Net. 인코더 2-level, 디코더 2-level, skip connection.
    파라미터 ~0.6M (ACA_CRNet 본체 대비 1% 미만).
    """

    def __init__(self, sar_channels: int = 2, opt_channels: int = 13,
                 base: int = 32):
        super().__init__()
        # Encoder
        self.enc0 = _conv_bn_relu(sar_channels, base)           # H
        self.enc1 = _conv_bn_relu(base, base * 2, s=2)          # H/2
        self.enc2 = _conv_bn_relu(base * 2, base * 4, s=2)      # H/4

        self.bottleneck = nn.Sequential(
            _conv_bn_relu(base * 4, base * 4),
            _conv_bn_relu(base * 4, base * 4),
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec1 = _conv_bn_relu(base * 4, base * 2)           # skip concat
        self.up0 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec0 = _conv_bn_relu(base * 2, base)               # skip concat

        # Head: 13ch pseudo-optical, Sigmoid (입력 normalize 범위 [0,1]과 일치)
        self.head = nn.Sequential(
            nn.Conv2d(base, opt_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, sar: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(sar)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d1 = self.dec1(torch.cat([self.up1(b), e1], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), e0], dim=1))
        return self.head(d0)  # (B, 13, H, W), in [0,1]


class SDIInjector(nn.Module):
    """SDI residual 주입 헤드.

    pred = cloudy + tail_out + λ · gate · pseudo_opt
    - λ: 학습 가능한 스칼라 (zero-init → 학습 초기 identity)
    - gate: (옵션) CAFM density map 재활용, 구름 영역 한정 주입
    """

    def __init__(self, use_density_gate: bool = True):
        super().__init__()
        self.lam = nn.Parameter(torch.zeros(1))  # zero-init
        self.use_density_gate = use_density_gate

    def forward(
        self,
        base_pred: torch.Tensor,            # (B,13,H,W) = cloudy + tail_out
        pseudo_opt: torch.Tensor,            # (B,13,H,W)
        density: torch.Tensor | None = None  # (B,1,H,W) ∈ [0,1]
    ) -> torch.Tensor:
        if self.use_density_gate and density is not None:
            injection = self.lam * density * pseudo_opt
        else:
            injection = self.lam * pseudo_opt
        return base_pred + injection
```

### 변경 파일: `modules/model/cafm/ACA_CRNet.py`

**`__init__` 추가** (기존 `use_cafm` 플래그 옆에):
```python
def __init__(self, ..., use_cafm=True, use_sdi=True, cafm_feat_dim=32):
    ...
    self.use_sdi = use_sdi
    if use_sdi:
        from .sdi import SARTranslator, SDIInjector
        self.sar_translator = SARTranslator(sar_channels, opt_channels)
        self.sdi_injector = SDIInjector(use_density_gate=use_cafm)
```

**`forward` 변경** (말미):
```python
# 기존
out = self.tail(feat)
pred = cloudy + out

# 변경
out = self.tail(feat)
base_pred = cloudy + out
if self.use_sdi:
    pseudo_opt = self.sar_translator(sar)
    # 시각화/보조 손실을 위해 보존
    self.last_pseudo_opt = pseudo_opt
    pred = self.sdi_injector(base_pred, pseudo_opt, density=self.last_density)
else:
    pred = base_pred
return pred
```

### 변경 파일: `tmp_main.py`

```python
parser.add_argument("--no-sdi", action="store_true",
                    help="SDI 모듈 없이 학습")

# build_model 내부
model = ACA_CRNet(
    sar_channels=2, opt_channels=13,
    num_layers=16, feature_sizes=256,
    use_cafm=not args.no_cafm,
    use_sdi=not args.no_sdi,
    cafm_feat_dim=32,
)
```

---

## 6. 학습 스펙

| 항목 | 값 |
|---|---|
| 추가 파라미터 | ~600K (SARTranslator) + 1 scalar |
| 추가 FLOPs | U-Net forward 1회 (base=32 기준 경미) |
| VRAM 추가 | ~1-2% (feature map 작음) |
| 학습률 | 기존 1e-4 유지 (λ zero-init이라 안정) |
| 배치/에폭 | 기존과 동일 (batch=4, 20 epochs) |

---

## 7. 보조 손실 옵션 (Option C로 확장 시)

`CloudAdaptiveLoss` 확장:
```python
# 주 손실: pred vs target
L_main = CloudAdaptiveLoss(pred, target)

# 보조 손실: pseudo_opt이 clean 영역에서 target과 일치하도록
#   구름 없는 영역(density ≈ 0)에서만 supervision
clean_mask = 1.0 - density   # (B,1,H,W)
L_trans = (clean_mask * (pseudo_opt - target).abs()).mean()

L_total = L_main + λ_aux * L_trans   # λ_aux = 0.1 권장
```

이를 통해 translator가 **"clean 영역에서 target과 일치하는 13ch HSI를 생성"** 하는 방향으로 학습됨 → 구름 영역 주입 품질 간접 향상.

---

## 8. Ablation 플랜

| 설정 | use_cafm | use_sdi | 보조손실 | 목적 |
|---|---|---|---|---|
| baseline | ✗ | ✗ | ✗ | 기준선 |
| +CAFM | ✓ | ✗ | ✗ | 현재 best |
| +SDI | ✗ | ✓ | ✗ | SDI 단독 효과 |
| +CAFM+SDI | ✓ | ✓ | ✗ | **Option B (권장 1순위)** |
| +CAFM+SDI+aux | ✓ | ✓ | ✓ | **Option C (확장)** |

**비교 메트릭**: PSNR, SSIM, SAM, MAE (기존 4개 유지).
예상 흐름:
- baseline PSNR 27.82 → +CAFM 28.50
- +CAFM+SDI 목표: **PSNR +0.5~1.0 dB (29.0~29.5)**, SSIM/SAM 동반 개선

---

## 9. 구현 체크리스트

- [ ] `modules/model/cafm/sdi.py` 신규 작성 (`SARTranslator`, `SDIInjector`)
- [ ] `modules/model/cafm/ACA_CRNet.py` 수정: `use_sdi` 플래그, forward 말미 변경
- [ ] `modules/model/cafm/__init__.py` 에 export 추가
- [ ] `tmp_main.py`: `--no-sdi` 플래그 추가
- [ ] `enable_gradient_checkpointing` 호환성 확인 (SARTranslator는 checkpoint 불필요)
- [ ] `λ` zero-init 확인 (학습 초기 기존 모델과 동일 출력)
- [ ] (Option C) `CloudAdaptiveLoss`에 auxiliary term 추가
- [ ] 20 epoch ablation 실행: baseline / +CAFM / +CAFM+SDI

---

## 10. 리스크 & 대응

| 리스크 | 대응 |
|---|---|
| Translator가 2048 샘플로 수렴 못함 | Option C(보조 손실) 적용, 또는 translator 더 얕게(base=16) |
| VRAM 증가로 OOM | SARTranslator를 `torch.utils.checkpoint`로 감싸기 |
| λ가 학습 중 음수로 수렴 | `λ = relu(lam_raw)` 또는 `softplus` 적용 |
| CAFM density가 부정확해 gating이 역효과 | `use_density_gate=False`로 단순 주입 테스트 |
| SDI가 과도하게 주입하여 출력 blur | `clamp(λ, 0, 0.3)` 또는 spectral consistency loss 추가 |

---

## 11. 참고

- **논문 PDF**: `docs/2025_TGRS_CHVo.pdf` (Sec III-B eq.(8), Table IV, Fig. 3)
- **저자 코드**: `docs/SAR-ULRTC-CR/main_net.py:11037-11091` (`RPCA_Net_4_stages_SAR_Trans_RGB_8`)
- **저자 데이터 로더**: `docs/SAR-ULRTC-CR/dataLoader.py:2060-2170` (오프라인 `trans` 필드 로딩)
- **관련 선행 연구**:
  - Meraner et al., *"Cloud removal in Sentinel-2 imagery using DSen2-CR"*, ISPRS J. 2020 (SAR residual 주입 원조)
  - Xu et al., *"AFR-CR: Adaptive Frequency Domain Reconstruction for SAR-Assisted Cloud Removal"*, Remote Sensing 2026

---

**다음 스텝**: `sdi.py` 작성 → `ACA_CRNet.py` 통합 → smoke test (train-max-samples 64, max-epochs 2) → 정규 20 epoch run.
